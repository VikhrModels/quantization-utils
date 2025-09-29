import argparse
import glob
import json
import logging
import os
import re
import select
import subprocess
import sys
import timeit
import shlex
from io import TextIOWrapper
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from modules.imatrix import Imatrix
from shared import get_environment_info, get_perplexity_command
from tabulate import tabulate

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("perplexity.log")],
)

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def extract_ppl_and_drift(output: List[str]) -> Optional[Tuple[float, float]]:
    for line in output[-50:]:
        match = re.search(r"Final estimate: PPL = ([\d\.]+) \+/- ([\d\.]+)", line)
        if match:
            return (float(match.group(1)), float(match.group(2)))
    return None


def handle_output(
    stream: Any,
    prefix: str,
    stdout_buffer: List[str],
    stderr_buffer: List[str],
    quiet: bool,
) -> None:
    for line in stream:
        if not quiet:
            print(f"{prefix}: {line}", end="", flush=True)
        if prefix == "STDOUT":
            stdout_buffer.append(line)
        elif "ETA" in line:
            logging.info(line.strip())
        stderr_buffer.append(line)


def stream_output(
    stream: TextIOWrapper, prefix: str, quiet: bool
) -> Generator[Tuple[str, str], None, None]:
    for line in iter(stream.readline, ""):
        if not quiet:
            print(f"{prefix}: {line}", end="", flush=True)
        yield (prefix.lower(), line)


def run_perplexity_command(
    command: str, args: argparse.Namespace
) -> Generator[Tuple[str, Union[str, float, Exception]], None, None]:
    start_time = timeit.default_timer()

    try:
        with subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            text=True,
            bufsize=1,
        ) as process:
            # Set up non-blocking reads
            for fd in (process.stdout, process.stderr):
                os.set_blocking(fd.fileno(), False)

            while process.poll() is None:
                ready, _, _ = select.select(
                    [process.stdout, process.stderr], [], [], 0.1
                )
                for stream in ready:
                    output = stream.read()
                    if output:
                        prefix = "STDOUT" if stream == process.stdout else "STDERR"
                        if not args.quiet:
                            print(f"{prefix}: {output}", end="", flush=True)
                        yield (prefix.lower(), output)

            # Read any remaining output
            for stream, prefix in [
                (process.stdout, "STDOUT"),
                (process.stderr, "STDERR"),
            ]:
                output = stream.read()
                if output:
                    if not args.quiet:
                        print(f"{prefix}: {output}", end="", flush=True)
                    yield (prefix.lower(), output)

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)

        end_time = timeit.default_timer()
        execution_time = end_time - start_time
        yield ("execution_time", execution_time)

    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with error code {e.returncode}")
        yield ("error", e)
    except OSError as e:
        logging.error(f"OS error occurred: {e}")
        yield ("error", e)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        yield ("error", e)


def extract_model_quant_id(file: str) -> str:
    basename = os.path.basename(file)
    model_quant_id = re.sub(r"^.*?-((?:[FIQ]|BF)[\d_XLQMS]+)", r"\1", basename)
    return model_quant_id.replace(".gguf", "")


def calculate_perplexity(
    model_name: str,
    stats_file: str,
    file: str,
    args: argparse.Namespace,
    dataset_name: str,
    dataset_path: str,
    force: bool = False,
) -> Dict[str, Any]:
    model_quant_id = extract_model_quant_id(file)
    model_results: Dict[str, Any] = {}

    loaded_data: Dict[str, Any] = {}
    if os.path.exists(stats_file) and os.path.getsize(stats_file) > 0:
        with open(stats_file, "r") as f:
            try:
                loaded_data = json.load(f)
            except json.JSONDecodeError:
                logging.error(
                    f"Error reading JSON data from {stats_file}. Starting with an empty dataset."
                )
                loaded_data = {}

    if (
        model_quant_id in loaded_data.get("perplexity", {}).get(dataset_name, {})
        and not force
    ):
        logging.info(f"Skipping {model_name} - already calculated")
        return loaded_data["perplexity"][dataset_name][model_quant_id]

    perplexity_cmd = get_perplexity_command()
    env_info = get_environment_info()

    command_parts: List[str] = [
        perplexity_cmd,
        "-m",
        file,
        "-f",
        dataset_path,
        "-t",
        str(getattr(args, "threads", os.cpu_count() or 1)),
    ]

    if env_info.get("acceleration") != "cpu":
        ngl_value = getattr(args, "ngl", None)
        if ngl_value is not None:
            command_parts.extend(["-ngl", str(ngl_value)])
        command_parts.append("-fa")

    command = " ".join(shlex.quote(part) for part in command_parts)

    output = []
    execution_time = None
    for output_type, content in run_perplexity_command(command, args):
        if output_type == "error":
            logging.error(content)
            return model_results
        elif output_type == "execution_time":
            execution_time = content
        elif output_type == "stdout":
            # in case if stdout
            logging.info(content)
            output.append(content)
        else:
            # in case if stderr
            logging.error(content)
            output.append(content)

    if output is None:
        logging.error(
            f"Failed to run perplexity command for {model_name} on {dataset_name}"
        )
        return model_results

    ppl_and_drift = extract_ppl_and_drift(output)
    if ppl_and_drift is None:
        logging.error(
            f"Failed to extract perplexity value or drift for {model_name} on {dataset_name}"
        )
        return model_results

    ppl, drift = ppl_and_drift
    model_results["perplexity"] = {"value": ppl, "drift": drift}
    model_results["time"] = execution_time

    human_readable_time = f"{execution_time:.2f}s"
    if execution_time >= 60:
        minutes, seconds = divmod(execution_time, 60)
        human_readable_time = f"{int(minutes)}m {seconds:.2f}s"
    logging.info(
        f"{model_name} - {dataset_name}: PPL = {ppl}, Time: {human_readable_time}"
    )

    return model_results


def run_perplexity(model_id: str, args: argparse.Namespace) -> Dict[str, Any]:
    if not model_id:
        raise ValueError("model_id is required")

    wikitext_path = "wikitext-2-raw/wiki.test.raw"
    if not os.path.exists(wikitext_path):
        raise FileNotFoundError(
            f"Wikitext-2 file not found: {wikitext_path}, use llama.cpp/scripts/get-wikitext-2.sh to download"
        )

    imatrix = Imatrix(model_id=model_id)

    # Prepare a test dataset if it doesn't exist.
    test_filepath = f"imatrix/{model_id}.test.txt"
    if not os.path.exists(test_filepath):
        dataset = imatrix.load_dataset()
        slice = imatrix.select_data_slice(dataset, qty=5000)
        with open(test_filepath, "w") as f:
            count = 0
            for example in slice:
                if example["conversations"]:
                    for conv in example["conversations"]:
                        f.write(conv["value"] or conv["original"] + "\n\n---\n\n")
                    count += 1
                    if count >= 500:
                        break
        logging.info(f"Wrote {count} examples to {test_filepath}")

    logging.info(f"Calculating perplexity of {model_id}")

    model_dir = os.path.join("models", f"{model_id}-GGUF")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Directory {model_dir} does not exist")

    gguf_files = glob.glob(os.path.join(model_dir, "*.gguf"))

    if not gguf_files:
        raise FileNotFoundError(f"No .gguf files found in {model_dir}")

    logging.info(f"Found {len(gguf_files)} .gguf files in {model_dir}:")

    results: Dict[str, Dict[str, Any]] = {"perplexity": {}, "time": {}}
    stats_file = os.path.join(model_dir, "stats.json")
    datasets = [("veles_sample_500", test_filepath), ("wikitext2", wikitext_path)]

    # Create the stats file if it doesn't exist
    if not os.path.exists(stats_file):
        open(stats_file, "a").close()

    for file in gguf_files:
        logging.info(f"  - {os.path.basename(file)}")
        model_name = os.path.basename(file).replace(".gguf", "")

        for dataset_name, dataset_path in datasets:
            # Read the current state
            with open(stats_file, "r") as json_file:
                try:
                    results = json.load(json_file)
                except json.JSONDecodeError:
                    results = {"perplexity": {}, "time": {}}

            # Check if we need to calculate perplexity for this model and dataset
            if (
                model_name in results.get("perplexity", {}).get(dataset_name, {})
                and not args.force_perplexity
            ):
                logging.info(
                    f"Skipping {model_name} - {dataset_name} - already calculated"
                )
                continue

            model_results = calculate_perplexity(
                model_name=model_name,
                stats_file=stats_file,
                file=file,
                args=args,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                force=args.force_perplexity,
            )

            for key in model_results:
                results.setdefault(key, {}).setdefault(dataset_name, {})[model_name] = (
                    model_results[key]
                )

            # Write updated results back to the file
            with open(stats_file, "w") as json_file:
                json.dump(results, json_file, indent=2)

            logging.info(f"Updated results saved to {stats_file}")

    logging.info(f"Results saved to {stats_file}")

    def print_results_table(results_key: str, title: str) -> None:
        table_data = []
        for model, data in results["perplexity"][results_key].items():
            perplexity = data["value"]
            drift = data["drift"]

            # Calculate lower and upper bounds
            lower_bound = perplexity - drift
            upper_bound = perplexity + drift

            table_data.append(
                [
                    model,
                    f"{perplexity:.4f}",
                    f"{lower_bound:.4f}",
                    f"{upper_bound:.4f}",
                    f"{drift:.4f}",
                ]
            )

        # Sort the table data based on the perplexity value (second column)
        table_data.sort(key=lambda x: float(x[1]))

        if table_data:
            headers = [
                "Model",
                "Perplexity",
                "Lower Bound",
                "Upper Bound",
                "Confidence",
            ]
            table = tabulate(table_data, headers=headers, tablefmt="pipe")
            print(f"\n{title}:")
            print(table)
        else:
            print(f"\nNo {title} available.")

    print_results_table("veles_sample_500", "Veles Results")
    print_results_table("wikitext2", "Wikitext-2 Results")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate perplexity of a quantized model"
    )
    parser.add_argument(
        "--quiet", "-q", help="Suppress output other than results", action="store_true"
    )
    parser.add_argument(
        "-ngl", type=int, help="Number of GPU layers to use", default=999
    )
    parser.add_argument(
        "--model-id", "-m", type=str, help="Huggingface model ID", required=True
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        help="Number of threads to use",
        default=os.cpu_count(),
    )
    parser.add_argument(
        "--force-perplexity",
        "-f",
        action="store_true",
        help="Force calculation of perplexity",
        default=False,
    )

    args = parser.parse_args()
    run_perplexity(args.model_id, args)

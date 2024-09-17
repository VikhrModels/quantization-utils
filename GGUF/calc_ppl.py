import argparse
import os
from modules.perplexity import run_perplexity

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Log the change of directory (optional, but helpful for debugging)
print(f"Changed working directory to: {os.getcwd()}")


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

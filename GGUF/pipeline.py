import argparse
import logging
import os
from typing import List

from modules.convert import Convert
from modules.imatrix import Imatrix
from modules.quantize import Quant, Quantize

os.chdir(os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s]: %(message)s"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GGUF Pipeline")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Vikhrmodels/Vikhr-Gemma-2B-instruct",
        help="Huggingface model ID",
        # required=True,
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Huggingface token, in case gated model is used",
        default=os.getenv("HF_TOKEN"),
    )
    parser.add_argument(
        "--quants",
        type=List[Quant],
        help="Quantization levels to run the pipeline on, default is every quantization level",
        default=[
            Quant.IQ1_S,
            Quant.IQ1_M,
            Quant.IQ2_XXS,
            Quant.IQ2_XS,
            Quant.IQ2_S,
            Quant.IQ2_M,
            Quant.Q2_K,
            Quant.Q2_K_S,
            Quant.IQ3_XXS,
            Quant.IQ3_XS,
            Quant.IQ3_S,
            Quant.IQ3_M,
            Quant.Q3_K,
            Quant.Q3_K_S,
            Quant.Q3_K_M,
            Quant.Q3_K_L,
            Quant.IQ4_NL,
            Quant.IQ4_XS,
            Quant.Q4_0,
            Quant.Q4_1,
            Quant.Q4_K,
            Quant.Q4_K_S,
            Quant.Q4_K_M,
            Quant.Q5_0,
            Quant.Q5_1,
            Quant.Q5_K,
            Quant.Q5_K_S,
            Quant.Q5_K_M,
            Quant.Q6_K,
            Quant.Q8_0,
            # Quant.F16,
            # Quant.BF16,
            # Quant.F32,
        ],
    )
    parser.add_argument(
        "--quants-skip",
        type=List[Quant],
        help="Quantization levels to skip for the pipeline",
        default=[],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force the whole pipeline to run",
        default=False,
    )
    parser.add_argument(
        "--force-imatrix-dataset",
        action="store_true",
        help="Force recreation of the imatrix dataset",
        default=False,
    )
    parser.add_argument(
        "--force-imatrix-calculation",
        action="store_true",
        help="Force calculation of the imatrix",
        default=False,
    )
    parser.add_argument(
        "--force-model-convert",
        action="store_true",
        help="Force conversion of the model",
        default=False,
    )
    parser.add_argument(
        "--force-model-quantization",
        action="store_true",
        help="Force quantization of the model",
        default=False,
    )

    parser.add_argument(
        "--q4_0-variants",
        type=List[str],
        help="Add specific q4_0 variants as of Q4_0_4_4, Q4_0_4_8, Q4_0_8_8",
        default=[],
    )

    args = parser.parse_args()

    # Propagate force flags
    if args.force:
        args.force_imatrix_dataset = True
        args.force_imatrix_calculation = True
        args.force_model_convert = True
        args.force_model_quantization = True
    if args.force_model_quantization:
        args.force_model_convert = True
    if args.force_imatrix_calculation:
        args.force_imatrix_dataset = True

    imatrix = Imatrix(model_id=args.model_id)
    convert = Convert(model_id=args.model_id, token=args.hf_token)
    quantize = Quantize(model_id=args.model_id)

    dirty = False

    if args.force_imatrix_dataset or not (
        os.path.exists(imatrix.get_imatrix_dataset_filepath())
        or os.path.exists(imatrix.get_imatrix_filepath())
    ):
        dirty = True
        imatrix.prepare_imatrix_samples()

    if args.force_model_convert or not os.path.exists(
        convert.get_converted_model_filepath()
    ):
        dirty = True
        convert.convert_model()

    if args.force_imatrix_calculation or not os.path.exists(
        imatrix.get_imatrix_filepath()
    ):
        dirty = True
        imatrix.calculate_imatrix(base_quant=convert.dtype.to_quant())

    if quantize.quantize_model(
        args.quants,
        args.quants_skip,
        converted_model_filepath=convert.get_converted_model_filepath(),
        base_quant=convert.dtype.to_quant(),
        imatrix_filepath=imatrix.get_imatrix_filepath(),
        force=args.force_model_quantization,
        q4_0_variants=args.q4_0_variants,
    ):
        dirty = True

    if dirty:
        logging.info("Processing complete")
    else:
        logging.info("Nothing to do, consider using force flags")

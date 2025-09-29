import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from exceptions import QuantizationError
from git import List
from shared import LoggerMixin, ModelMixin, Quant, get_quantize_command, run_command


class Quantize(LoggerMixin, ModelMixin):
    def __init__(self, model_id: str, *args, **kwargs):
        super().__init__(model_id=model_id, *args, **kwargs)
        self.config = get_config().quantization

    def quantize_model(
        self,
        quants: List[Quant],
        quants_skip: List[Quant] = [],
        base_quant: Quant = Quant.F32,
        converted_model_filepath: str = None,
        imatrix_filepath: str = None,
        force: bool = False,
        q4_0_variants: List[str] = [],
    ):
        quantize_cmd = get_quantize_command()
        self.info(f"Using quantize binary: {quantize_cmd}")

        if base_quant == Quant.F16:
            for quant in [Quant.F16, Quant.BF16, Quant.F32]:
                if quant not in quants and quant not in quants_skip:
                    quants_skip.append(quant)
        quants = [quant for quant in quants if quant not in quants_skip]

        imatrix_attrs = [
            "--imatrix",
            imatrix_filepath,
        ]

        for quant in quants:
            if force or not os.path.exists(self.get_quantized_filepath(quant)):
                variants = [quant.value]
                self.info(f"Quantizing model {self.model_id} for quant {quant}")
                if quant == Quant.Q4_0 and len(q4_0_variants) > 0:
                    for q4_0_variant in q4_0_variants:
                        variants.append(q4_0_variant)

                for variant in variants:
                    try:
                        command = [
                            quantize_cmd,
                            *(
                                imatrix_attrs
                                if quant.value in self.config.imatrix_quants
                                else []
                            ),
                            converted_model_filepath,
                            self.get_quantized_filepath(quant),
                            variant,
                        ]
                        print(" ".join(command))

                        run_command(
                            self.logger,
                            command,
                            ".",
                        )
                    except Exception as e:
                        raise QuantizationError(
                            self.model_id, quant_type=quant.value, reason=str(e)
                        ) from e

        return True

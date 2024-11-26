import os

from git import List
from shared import LoggerMixin, ModelMixin, Quant, run_command


QUANTIZE_CMD = "./llama-quantize"


class Quantize(LoggerMixin, ModelMixin):
    def __init__(self, model_id: str, *args, **kwargs):
        super().__init__(model_id=model_id, *args, **kwargs)

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
        # Prevent auto-skip for higher quants if they was explicitly requested
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
                    command = [
                        QUANTIZE_CMD,
                        # Do not use imatrix for upper quants, may lead to lower quality
                        *(
                            imatrix_attrs
                            if quant
                            in {
                                Quant.IQ1_M,
                                Quant.IQ1_S,
                                Quant.IQ2_XXS,
                                Quant.IQ2_XS,
                                Quant.IQ2_S,
                                Quant.IQ2_M,
                                Quant.IQ3_XXS,
                                Quant.IQ3_XS,
                                Quant.IQ3_S,
                                Quant.IQ3_M,
                                Quant.IQ4_NL,
                                Quant.IQ4_XS,
                                Quant.Q2_K_S,
                            }
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
                        "llama.cpp",
                    )

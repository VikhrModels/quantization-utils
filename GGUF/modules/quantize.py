import os
from git import List

from shared import LoggerMixin, ModelMixin, Quant, run_command

DTYPE_FP16 = "float16"
DTYPE_FP32 = "float32"
DTYPE_BF16 = "bfloat16"


class Quantize(LoggerMixin, ModelMixin):
    def __init__(self, model_id: str, *args, **kwargs):
        super().__init__(model_id=model_id, *args, **kwargs)

    def quantize_model(
        self,
        quants: List[Quant],
        quants_skip: List[Quant] = [],
        dtype: str = DTYPE_FP32,
        converted_model_filepath: str = None,
        imatrix_filepath: str = None,
        force: bool = False,
    ):
        # Prevent auto-skip for higher quants if they was explicitly requested
        if dtype == DTYPE_FP16:
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
                self.info(f"Quantizing model {self.model_id} for quant {quant}")
                run_command(
                    self.logger,
                    [
                    "./quantize",
                    # Do not use imatrix for upper quants, may lead to lower quality
                    *(
                        imatrix_attrs
                        if quant not in {
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
                            Quant.F16,
                            Quant.BF16,
                            Quant.F32,
                        }
                        else []
                    ),
                    converted_model_filepath,
                    self.get_quantized_filepath(quant),
                    quant.value,
                ],
                "llama.cpp",
            )

from enum import Enum


class Quant(Enum):
    """Quantization types supported by llama.cpp"""

    IQ1_S = "IQ1_S"
    IQ1_M = "IQ1_M"
    IQ2_XXS = "IQ2_XXS"
    IQ2_XS = "IQ2_XS"
    IQ2_S = "IQ2_S"
    IQ2_M = "IQ2_M"
    Q2_K = "Q2_K"
    Q2_K_S = "Q2_K_S"
    IQ3_XXS = "IQ3_XXS"
    IQ3_S = "IQ3_S"
    IQ3_M = "IQ3_M"
    Q3_K = "Q3_K"
    IQ3_XS = "IQ3_XS"
    Q3_K_S = "Q3_K_S"
    Q3_K_M = "Q3_K_M"
    Q3_K_L = "Q3_K_L"
    IQ4_NL = "IQ4_NL"
    IQ4_XS = "IQ4_XS"
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q4_K = "Q4_K"
    Q4_K_S = "Q4_K_S"
    Q4_K_M = "Q4_K_M"
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q5_K = "Q5_K"
    Q5_K_S = "Q5_K_S"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    F16 = "F16"
    BF16 = "BF16"
    F32 = "F32"


class DType(Enum):
    """Data types for model conversion"""

    FP16 = "float16"
    FP32 = "float32"
    BF16 = "bfloat16"

    def to_quant(self) -> Quant:
        """Convert DType to corresponding Quant enum"""
        if self.value == "float16":
            return Quant.F16
        elif self.value == "float32":
            return Quant.F32
        elif self.value == "bfloat16":
            return Quant.BF16
        else:
            raise ValueError(f"Invalid dtype: {self}")

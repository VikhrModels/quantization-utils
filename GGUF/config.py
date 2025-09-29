import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class IMatrixConfig:
    """Configuration for importance matrix calculation"""

    default_dataset: str = "Vikhrmodels/Veles-2.5"
    sample_qty: int = 1000
    context_size: int = 512
    batch_size: int = 512
    chunks: int = 1024
    gpu_layers: int = 32
    temperature: float = 0.25
    calibration_data_dir: str = "resources/standard_cal_data"

    def __post_init__(self):
        if self.sample_qty < 1:
            raise ValueError("sample_qty must be positive")
        if self.context_size < 1:
            raise ValueError("context_size must be positive")


@dataclass
class PerplexityConfig:
    """Configuration for perplexity calculation"""

    sample_size: int = 5000
    max_examples: int = 500
    default_ngl: int = 999
    default_threads: Optional[int] = None
    wikitext_path: str = "wikitext-2-raw/wiki.test.raw"
    enable_flash_attn: bool = True

    def __post_init__(self):
        if self.default_threads is None:
            self.default_threads = os.cpu_count() or 1


@dataclass
class PathConfig:
    """Configuration for file paths"""

    base_dir: Path = field(default_factory=lambda: Path.cwd())
    models_dir: str = "models"
    output_dir: str = "output"
    imatrix_dir: str = "imatrix"
    resources_dir: str = "resources"

    def get_models_path(self) -> Path:
        return self.base_dir / self.models_dir

    def get_output_path(self) -> Path:
        return self.base_dir / self.output_dir

    def get_imatrix_path(self) -> Path:
        return self.base_dir / self.imatrix_dir

    def get_resources_path(self) -> Path:
        return self.base_dir / self.resources_dir


@dataclass
class LlamaBinaryConfig:
    """Configuration for llama.cpp binaries"""

    search_paths: List[str] = field(
        default_factory=lambda: [
            str(Path.home() / ".local" / "bin"),
            "/usr/local/bin",
            "/opt/homebrew/bin",
            "llama.cpp/build/bin",
            "llama.cpp",
        ]
    )
    required_binaries: List[str] = field(
        default_factory=lambda: [
            "llama-cli",
            "llama-quantize",
            "llama-perplexity",
            "llama-imatrix",
        ]
    )


@dataclass
class NetworkConfig:
    """Configuration for network operations"""

    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    timeout: int = 300

    def __post_init__(self):
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")


@dataclass
class QuantizationConfig:
    """Configuration for quantization process"""

    imatrix_quants: List[str] = field(
        default_factory=lambda: [
            "IQ1_M",
            "IQ1_S",
            "IQ2_XXS",
            "IQ2_XS",
            "IQ2_S",
            "IQ2_M",
            "IQ3_XXS",
            "IQ3_XS",
            "IQ3_S",
            "IQ3_M",
            "IQ4_NL",
            "IQ4_XS",
            "Q2_K_S",
        ]
    )
    default_quants: List[str] = field(
        default_factory=lambda: [
            "IQ1_S",
            "IQ1_M",
            "IQ2_XXS",
            "IQ2_XS",
            "IQ2_S",
            "IQ2_M",
            "Q2_K",
            "Q2_K_S",
            "IQ3_XXS",
            "IQ3_XS",
            "IQ3_S",
            "IQ3_M",
            "Q3_K",
            "Q3_K_S",
            "Q3_K_M",
            "Q3_K_L",
            "IQ4_NL",
            "IQ4_XS",
            "Q4_0",
            "Q4_1",
            "Q4_K",
            "Q4_K_S",
            "Q4_K_M",
            "Q5_0",
            "Q5_1",
            "Q5_K",
            "Q5_K_S",
            "Q5_K_M",
            "Q6_K",
            "Q8_0",
        ]
    )


@dataclass
class Config:
    """Main configuration container"""

    imatrix: IMatrixConfig = field(default_factory=IMatrixConfig)
    perplexity: PerplexityConfig = field(default_factory=PerplexityConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    binaries: LlamaBinaryConfig = field(default_factory=LlamaBinaryConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables"""
        config = cls()

        if hf_token := os.getenv("HF_TOKEN"):
            os.environ["HF_TOKEN"] = hf_token

        if threads := os.getenv("OMP_NUM_THREADS"):
            try:
                config.perplexity.default_threads = int(threads)
            except ValueError:
                pass

        if base_dir := os.getenv("QUANTIZATION_BASE_DIR"):
            config.paths.base_dir = Path(base_dir)

        return config


_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = Config.from_env()
    return _global_config


def set_config(config: Config) -> None:
    """Set global configuration instance"""
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset configuration to default"""
    global _global_config
    _global_config = None

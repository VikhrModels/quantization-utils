class QuantizationUtilsError(Exception):
    """Base exception for all quantization-utils errors"""

    pass


class BinaryNotFoundError(QuantizationUtilsError):
    """Raised when required llama.cpp binary is not found"""

    def __init__(self, binary_name: str, searched_paths: list = None):
        self.binary_name = binary_name
        self.searched_paths = searched_paths or []
        message = (
            f"llama.cpp binary '{binary_name}' not found. "
            f"Please run 'python setup.py' to install llama.cpp or ensure it's in your PATH."
        )
        if searched_paths:
            message += f"\nSearched in: {', '.join(searched_paths)}"
        super().__init__(message)


class ConversionScriptNotFoundError(QuantizationUtilsError):
    """Raised when conversion script is not found"""

    def __init__(self, script_name: str):
        self.script_name = script_name
        message = (
            f"{script_name} script not found. "
            f"Please ensure llama.cpp is properly installed or run 'python setup.py --force-rebuild'."
        )
        super().__init__(message)


class CommandExecutionError(QuantizationUtilsError):
    """Raised when external command execution fails"""

    def __init__(self, command: str, return_code: int, cwd: str = "."):
        self.command = command
        self.return_code = return_code
        self.cwd = cwd
        message = f"Command failed with exit code {return_code}: {command} (in {cwd})"
        super().__init__(message)


class ModelDownloadError(QuantizationUtilsError):
    """Raised when model download fails"""

    def __init__(self, model_id: str, reason: str = None):
        self.model_id = model_id
        self.reason = reason
        message = f"Failed to download model '{model_id}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class ModelConversionError(QuantizationUtilsError):
    """Raised when model conversion fails"""

    def __init__(self, model_id: str, reason: str = None):
        self.model_id = model_id
        self.reason = reason
        message = f"Failed to convert model '{model_id}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class QuantizationError(QuantizationUtilsError):
    """Raised when quantization fails"""

    def __init__(self, model_id: str, quant_type: str = None, reason: str = None):
        self.model_id = model_id
        self.quant_type = quant_type
        self.reason = reason
        message = f"Failed to quantize model '{model_id}'"
        if quant_type:
            message += f" to {quant_type}"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class IMatrixCalculationError(QuantizationUtilsError):
    """Raised when imatrix calculation fails"""

    def __init__(self, model_id: str, reason: str = None):
        self.model_id = model_id
        self.reason = reason
        message = f"Failed to calculate imatrix for model '{model_id}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class PerplexityCalculationError(QuantizationUtilsError):
    """Raised when perplexity calculation fails"""

    def __init__(self, model_name: str, dataset: str = None, reason: str = None):
        self.model_name = model_name
        self.dataset = dataset
        self.reason = reason
        message = f"Failed to calculate perplexity for '{model_name}'"
        if dataset:
            message += f" on dataset '{dataset}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class EnvironmentValidationError(QuantizationUtilsError):
    """Raised when environment validation fails"""

    def __init__(self, missing_components: list = None, reason: str = None):
        self.missing_components = missing_components or []
        self.reason = reason
        message = "Environment validation failed"
        if missing_components:
            message += f": missing {', '.join(missing_components)}"
        if reason:
            message += f" - {reason}"
        super().__init__(message)


class ConfigurationError(QuantizationUtilsError):
    """Raised when configuration is invalid"""

    def __init__(self, field: str = None, reason: str = None):
        self.field = field
        self.reason = reason
        message = "Invalid configuration"
        if field:
            message += f" for field '{field}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class NetworkError(QuantizationUtilsError):
    """Raised when network operations fail"""

    def __init__(self, operation: str, reason: str = None, retry_count: int = 0):
        self.operation = operation
        self.reason = reason
        self.retry_count = retry_count
        message = f"Network operation failed: {operation}"
        if retry_count > 0:
            message += f" (after {retry_count} retries)"
        if reason:
            message += f" - {reason}"
        super().__init__(message)

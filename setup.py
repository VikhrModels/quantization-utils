#!/usr/bin/env python3
"""
Quantization Utils Setup Script
Initializes environment for bare metal installation with OS detection and llama.cpp setup
"""

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def detect_os():
    """Detect operating system and architecture."""

    system = platform.system().lower()
    machine = platform.machine().lower()

    # Normalize architecture names
    if machine in ["x86_64", "amd64"]:
        arch = "x86_64"
    elif machine in ["aarch64", "arm64"]:
        arch = "arm64"
    else:
        arch = machine

    return system, arch


def has_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def has_metal():
    """Check if Metal is available (macOS)."""
    system, _ = detect_os()
    return system == "darwin"


def get_acceleration_type():
    """Determine the best acceleration type available."""
    if has_nvidia_gpu():
        return "cuda"
    elif has_metal():
        return "metal"
    else:
        return "cpu"


def get_latest_release_info():
    """Get information about the latest llama.cpp release."""
    url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                url, headers={"User-Agent": "quantization-utils/1.0"}
            )
        ) as response:
            data = json.loads(response.read())
            return data["tag_name"], data["assets"]
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        logger.error(f"Failed to get release info: {e}")
        return None, None


def find_best_binary(assets, system, arch, acceleration):
    """Find the best binary for the current system."""
    # Priority order for different acceleration types
    if system == "linux":
        if acceleration == "cuda":
            # No pre-built CUDA binaries for Linux, need to build from source
            return None
        elif acceleration == "cpu":
            candidates = [
                "llama-{}-bin-ubuntu-x64.zip",
                "llama-{}-bin-linux-x64.zip",
            ]
    elif system == "darwin":
        if arch == "arm64":
            candidates = ["llama-{}-bin-macos-arm64.zip"]
        else:
            candidates = ["llama-{}-bin-macos-x64.zip"]
    elif system == "windows":
        if acceleration == "cuda":
            candidates = [
                "llama-{}-bin-win-cuda-12.4-x64.zip",
                "llama-{}-bin-win-cuda-12.2-x64.zip",
                "llama-{}-bin-win-cuda-12.1-x64.zip",
                "llama-{}-bin-win-cuda-11.8-x64.zip",
                "llama-{}-bin-win-cuda-x64.zip",
            ]
        else:
            candidates = ["llama-{}-bin-win-cpu-x64.zip"]
    else:
        return None

    # Find the first matching asset
    for asset in assets:
        name = asset["name"]
        for candidate_pattern in candidates:
            # Extract tag from asset name pattern
            if "llama-" in name and "-bin-" in name:
                parts = name.split("-")
                if len(parts) >= 3:
                    tag = parts[
                        1
                    ]  # e.g., 'b5773' from 'llama-b5773-bin-ubuntu-x64.zip'
                    candidate = candidate_pattern.format(tag)
                    if name == candidate:
                        return asset["browser_download_url"]

    return None


def download_and_extract_binary(url, install_dir):
    """Download and extract llama.cpp binary."""
    logger.info(f"Trying to download: {url}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Download
        zip_path = Path(temp_dir) / "llama.zip"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    url, headers={"User-Agent": "quantization-utils/1.0"}
                )
            ) as response:
                with open(zip_path, "wb") as f:
                    f.write(response.read())
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            logger.error(f"Failed to download {url}: {e}")
            return False

        # Extract
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for member in zip_ref.namelist():
                    # Extract all files
                    zip_ref.extract(member, temp_dir)
                    src_path = Path(temp_dir) / member
                    if src_path.is_file():
                        # Make sure install directory exists
                        install_dir.mkdir(parents=True, exist_ok=True)
                        # Copy to install directory
                        dst_path = install_dir / src_path.name
                        shutil.copy2(src_path, dst_path)
                        # Make executable if it's a binary
                        if src_path.suffix == "" and src_path.name.startswith("llama"):
                            dst_path.chmod(0o755)
                        logger.info(f"Installed: {dst_path}")
        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
            return False

    return True


def build_from_source_with_cuda(fast_build=False):
    """Build llama.cpp from source with CUDA support."""
    if fast_build:
        logger.info("📦 Fast building llama.cpp from source with CUDA support...")
    else:
        logger.info("📦 Building llama.cpp from source with CUDA support...")

    build_dir = Path.home() / ".cache" / "quantization-utils" / "llama-build"
    install_dir = Path.home() / ".local" / "bin"

    try:
        # Clean previous build
        if build_dir.exists():
            shutil.rmtree(build_dir)

        build_dir.mkdir(parents=True, exist_ok=True)

        # Clone llama.cpp (shallow for fast build)
        clone_args = [
            "git",
            "clone",
            "https://github.com/ggerganov/llama.cpp.git",
            str(build_dir),
        ]

        if fast_build:
            clone_args.extend(["--depth", "1"])  # Shallow clone
            logger.info("Shallow cloning llama.cpp repository...")
        else:
            clone_args.extend(["--depth", "1"])  # Always use shallow for speed
            logger.info("Cloning llama.cpp repository...")

        subprocess.run(clone_args, check=True, capture_output=True)

        # Check if CUDA is available and determine the best CUDA to use
        cuda_available = False
        cuda_version = None
        nvcc_path = None

        # First try conda environment NVCC
        conda_nvcc = Path(os.environ.get("CONDA_PREFIX", "")) / "bin" / "nvcc"
        if conda_nvcc.exists():
            try:
                result = subprocess.run(
                    [str(conda_nvcc), "--version"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if "release 12." in result.stdout:  # CUDA 12.x supports compute_90
                    cuda_available = True
                    nvcc_path = str(conda_nvcc)
                    cuda_version = "conda"
                    logger.info(f"Using conda NVCC: {nvcc_path}")
                else:
                    logger.warning("Conda NVCC version is too old for H100 support")
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

        # If conda NVCC is unavailable or too old, try system NVCC
        if not cuda_available:
            system_paths = ["/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc"]
            for path in system_paths:
                if Path(path).exists():
                    try:
                        result = subprocess.run(
                            [path, "--version"],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        if (
                            "release 12." in result.stdout
                            or "release 11.8" in result.stdout
                        ):  # 11.8+ supports most modern GPUs
                            cuda_available = True
                            nvcc_path = path
                            cuda_version = "system"
                            logger.info(f"Using system NVCC: {nvcc_path}")
                            break
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        continue

        if not cuda_available:
            logger.warning("No suitable NVCC found - building with CPU support only")

        # Create build directory
        cmake_build_dir = build_dir / "build"
        cmake_build_dir.mkdir(exist_ok=True)

        # Configure with CMake
        cmake_args = [
            "cmake",
            "..",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]

        if fast_build:
            # Fast build optimizations
            cmake_args.extend(
                [
                    "-DLLAMA_BUILD_TESTS=OFF",  # Skip tests
                    "-DLLAMA_BUILD_EXAMPLES=OFF",  # Skip examples
                    "-DLLAMA_BUILD_SERVER=ON",  # Keep server
                    "-DBUILD_SHARED_LIBS=OFF",  # Static linking (faster)
                ]
            )

        if cuda_available:
            cmake_args.extend(["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_COMPILER={nvcc_path}"])

            # Set up CUDA environment for the build
            build_env = os.environ.copy()
            if cuda_version == "system":
                # Use system CUDA paths
                cuda_root = Path(nvcc_path).parent.parent
                build_env["CUDA_HOME"] = str(cuda_root)
                build_env["CUDA_ROOT"] = str(cuda_root)
                build_env["PATH"] = f"{cuda_root}/bin:" + build_env.get("PATH", "")
                build_env["LD_LIBRARY_PATH"] = (
                    f"{cuda_root}/lib64:{cuda_root}/lib:"
                    + build_env.get("LD_LIBRARY_PATH", "")
                )
            else:
                # Use conda CUDA paths
                conda_prefix = os.environ.get("CONDA_PREFIX", "")
                build_env["CUDA_HOME"] = conda_prefix
                build_env["CUDA_ROOT"] = conda_prefix
                build_env["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:" + build_env.get(
                    "LD_LIBRARY_PATH", ""
                )

            logger.info(f"Configuring with CUDA support using {cuda_version} NVCC...")
        else:
            build_env = os.environ.copy()
            logger.info("Configuring with CPU support...")

        subprocess.run(cmake_args, cwd=cmake_build_dir, check=True, env=build_env)

        # Build with optimizations
        build_cmd = ["cmake", "--build", ".", "--config", "Release"]

        if fast_build:
            # Use more parallel jobs for faster build
            import multiprocessing

            cores = multiprocessing.cpu_count()
            build_cmd.extend(
                ["-j", str(min(cores, 8))]
            )  # Cap at 8 to avoid memory issues
            logger.info(f"Fast building llama.cpp using {min(cores, 8)} cores...")
        else:
            build_cmd.extend(["-j"])
            logger.info("Building llama.cpp (this may take several minutes)...")

        subprocess.run(build_cmd, cwd=cmake_build_dir, check=True, env=build_env)

        # Install binaries and scripts
        logger.info("Installing binaries and conversion scripts...")
        install_dir.mkdir(parents=True, exist_ok=True)

        # Copy main binaries - they are built in build/bin/ subdirectory
        binaries = [
            "llama-cli",
            "llama-server",
            "llama-quantize",
            "llama-perplexity",
            "llama-imatrix",
            "llama-bench",
        ]

        installed_count = 0
        for binary in binaries:
            # Look in both build/ and build/bin/ directories
            src_paths = [cmake_build_dir / binary, cmake_build_dir / "bin" / binary]

            src_path = None
            for path in src_paths:
                if path.exists():
                    src_path = path
                    break

            if src_path and src_path.exists():
                dst_path = install_dir / binary
                shutil.copy2(src_path, dst_path)
                dst_path.chmod(0o755)
                logger.info(f"Installed: {dst_path}")
                installed_count += 1
            else:
                logger.warning(f"Binary not found: {binary}")

        # Copy Python conversion scripts
        conversion_scripts = [
            "convert_hf_to_gguf.py",
            "convert_legacy_llama.py",
            "convert.py",
        ]

        scripts_installed = 0
        for script in conversion_scripts:
            src_path = build_dir / script
            if src_path.exists():
                dst_path = install_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                dst_path.chmod(0o755)
                logger.info(f"Installed: {dst_path}")
                scripts_installed += 1

        # Copy gguf-py module if it exists
        gguf_src = build_dir / "gguf-py"
        if gguf_src.exists():
            gguf_dst = install_dir / "gguf-py"
            if gguf_dst.exists():
                shutil.rmtree(gguf_dst)
            shutil.copytree(gguf_src, gguf_dst)
            logger.info(f"Installed: {gguf_dst}")
            scripts_installed += 1

            # Add gguf-py to Python path by creating a .pth file
            try:
                import site

                site_packages = site.getsitepackages()
                if site_packages:
                    pth_file = Path(site_packages[0]) / "gguf_local.pth"
                    with open(pth_file, "w") as f:
                        f.write(str(gguf_dst))
                    logger.info(f"Added {gguf_dst} to Python path via {pth_file}")
            except Exception as e:
                logger.warning(f"Could not add gguf-py to Python path: {e}")

        # Also check for examples directory with conversion scripts
        examples_dir = build_dir / "examples"
        if examples_dir.exists():
            for script_file in examples_dir.glob("convert*.py"):
                dst_path = install_dir / script_file.name
                shutil.copy2(script_file, dst_path)
                dst_path.chmod(0o755)
                logger.info(f"Installed: {dst_path}")
                scripts_installed += 1

        if installed_count == 0:
            logger.error("No binaries were built successfully")
            return False

        logger.info(
            f"Successfully built and installed {installed_count} binaries and {scripts_installed} scripts with {'CUDA' if cuda_available else 'CPU'} support"
        )

        # Clean up build directory after successful installation
        if build_dir.exists():
            try:
                shutil.rmtree(build_dir)
                logger.info("Cleaned up build directory")
            except Exception as e:
                logger.warning(f"Failed to clean up build directory: {e}")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Build failed: {e}")
        # Clean up on failure
        if build_dir.exists():
            try:
                shutil.rmtree(build_dir)
            except Exception:
                pass
        return False
    except Exception as e:
        logger.error(f"Unexpected error during build: {e}")
        # Clean up on failure
        if build_dir.exists():
            try:
                shutil.rmtree(build_dir)
            except Exception:
                pass
        return False


def install_scripts_only():
    """Install only the conversion scripts without rebuilding binaries."""
    logger.info("📄 Installing conversion scripts only...")

    build_dir = Path.home() / ".cache" / "quantization-utils" / "llama-scripts"
    install_dir = Path.home() / ".local" / "bin"

    try:
        # Clean previous download
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        # Shallow clone just for scripts
        logger.info("Downloading conversion scripts...")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/ggerganov/llama.cpp.git",
                str(build_dir),
            ],
            check=True,
            capture_output=True,
        )

        # Copy conversion scripts
        conversion_scripts = [
            "convert_hf_to_gguf.py",
            "convert_legacy_llama.py",
            "convert.py",
        ]

        scripts_installed = 0
        install_dir.mkdir(parents=True, exist_ok=True)

        for script in conversion_scripts:
            src_path = build_dir / script
            if src_path.exists():
                dst_path = install_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                dst_path.chmod(0o755)
                logger.info(f"Installed: {dst_path}")
                scripts_installed += 1

        # Copy gguf-py module
        gguf_src = build_dir / "gguf-py"
        if gguf_src.exists():
            gguf_dst = install_dir / "gguf-py"
            if gguf_dst.exists():
                shutil.rmtree(gguf_dst)
            shutil.copytree(gguf_src, gguf_dst)
            logger.info(f"Installed: {gguf_dst}")
            scripts_installed += 1

        # Also check examples directory
        examples_dir = build_dir / "examples"
        if examples_dir.exists():
            for script_file in examples_dir.glob("convert*.py"):
                dst_path = install_dir / script_file.name
                shutil.copy2(script_file, dst_path)
                dst_path.chmod(0o755)
                logger.info(f"Installed: {dst_path}")
                scripts_installed += 1

        # Clean up
        if build_dir.exists():
            shutil.rmtree(build_dir)

        logger.info(f"✅ Installed {scripts_installed} conversion scripts")
        return True

    except Exception as e:
        logger.error(f"Failed to install scripts: {e}")
        if build_dir.exists():
            try:
                shutil.rmtree(build_dir)
            except Exception:
                pass
        return False


def install_llama_cpp(fast_build=False, scripts_only=False):
    """Install llama.cpp binaries and/or scripts."""
    if scripts_only:
        logger.info("📄 Installing conversion scripts only...")
        return install_scripts_only()

    system, arch = detect_os()
    acceleration = get_acceleration_type()

    logger.info("📦 Installing llama.cpp...")

    # For Linux with CUDA, we need to build from source
    if system == "linux" and acceleration == "cuda":
        logger.info(
            "🔧 CUDA detected on Linux - building from source (pre-built CUDA binaries not available)"
        )
        return build_from_source_with_cuda(fast_build=fast_build)

    # Try to download pre-built binaries
    tag, assets = get_latest_release_info()
    if not tag or not assets:
        logger.error("Failed to get release information")
        return False

    binary_url = find_best_binary(assets, system, arch, acceleration)
    if not binary_url:
        if system == "linux" and acceleration == "cpu":
            # Try building from source as fallback
            logger.warning("No suitable pre-built binary found - building from source")
            return build_from_source_with_cuda(fast_build=fast_build)
        else:
            logger.error(
                f"No suitable binary found for {system}-{arch} with {acceleration} acceleration"
            )
            return False

    install_dir = Path.home() / ".local" / "bin"

    success = download_and_extract_binary(binary_url, install_dir)
    if success:
        logger.info("Successfully installed llama.cpp binaries")
        # Also install scripts after downloading binaries
        if not install_scripts_only():
            logger.warning("Failed to install conversion scripts")

    return success


def setup_cuda_environment():
    """Set up CUDA environment variables in the conda environment."""
    logger.info("🔧 Setting up CUDA environment variables...")

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        logger.warning("CONDA_PREFIX not set - skipping environment variable setup")
        return

    # Check if conda has CUDA installed
    conda_nvcc = Path(conda_prefix) / "bin" / "nvcc"
    if conda_nvcc.exists():
        try:
            # Set conda CUDA environment variables
            env_vars = {
                "CUDA_HOME": conda_prefix,
                "CUDA_ROOT": conda_prefix,
                "LD_LIBRARY_PATH": f"{conda_prefix}/lib:${{LD_LIBRARY_PATH:-}}",
            }

            for var, value in env_vars.items():
                try:
                    subprocess.run(
                        ["conda", "env", "config", "vars", "set", f"{var}={value}"],
                        check=True,
                        capture_output=True,
                    )
                    logger.info(f"Set {var}={value}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to set {var}: {e}")

            logger.info("✅ CUDA environment variables configured")
            logger.info(
                "Please run: conda deactivate && conda activate quantization-utils"
            )

        except Exception as e:
            logger.warning(f"Failed to configure CUDA environment variables: {e}")
    else:
        logger.info("Conda NVCC not found - using system CUDA if available")


def setup_directories():
    """Create necessary directories."""
    logger.info("📁 Setting up directories...")

    base_dir = Path.cwd()
    directories = [
        base_dir / "GGUF" / "models",
        base_dir / "GGUF" / "output",
        base_dir / "GGUF" / "imatrix",
        base_dir / "GGUF" / "resources/standard_cal_data",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def validate_environment():
    """Validate that the environment is set up correctly."""
    logger.info("🔍 Validating environment...")

    # Check if binaries are available
    install_dir = Path.home() / ".local" / "bin"
    required_binaries = ["llama-cli", "llama-quantize", "llama-perplexity"]

    missing_binaries = []
    for binary in required_binaries:
        binary_path = install_dir / binary
        if not binary_path.exists():
            # Also check if it's in PATH
            try:
                subprocess.run([binary, "--help"], capture_output=True, check=True)
            except (FileNotFoundError, subprocess.CalledProcessError):
                missing_binaries.append(binary)

    if missing_binaries:
        logger.error(f"Missing binaries: {', '.join(missing_binaries)}")
        return False

    # Check acceleration
    system, _ = detect_os()
    acceleration = get_acceleration_type()

    if acceleration == "cuda":
        # Test CUDA support
        try:
            result = subprocess.run(
                [str(install_dir / "llama-cli"), "--help"],
                capture_output=True,
                text=True,
            )

            if "CUDA" in result.stderr or "cuda" in result.stderr.lower():
                logger.info("✅ CUDA acceleration is available")
            else:
                logger.warning(
                    "⚠️  CUDA was detected but may not be enabled in binaries"
                )

        except Exception as e:
            logger.warning(f"Could not verify CUDA support: {e}")

    logger.info("✅ Environment validation completed")
    return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Quantization Utils Environment")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the environment, do not install",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild from source even if binaries exist",
    )
    parser.add_argument(
        "--fast-build",
        action="store_true",
        help="Force fast build from source even if binaries exist",
    )
    parser.add_argument(
        "--scripts-only",
        action="store_true",
        help="Only install conversion scripts without rebuilding binaries",
    )
    args = parser.parse_args()

    logger.info("🚀 Starting Quantization Utils Setup")

    if args.validate_only:
        success = validate_environment()
        sys.exit(0 if success else 1)

    # Detect environment
    logger.info("🔍 Detecting environment...")
    system, arch = detect_os()
    has_nvidia = has_nvidia_gpu()
    has_metal_support = has_metal()
    acceleration = get_acceleration_type()

    logger.info("📋 Environment Summary:")
    logger.info(f"   OS: {system}")
    logger.info(f"   Architecture: {arch}")
    logger.info(f"   NVIDIA GPU: {has_nvidia}")
    logger.info(f"   Metal: {has_metal_support}")
    logger.info(f"   Acceleration: {acceleration}")

    # Setup CUDA environment if needed
    if acceleration == "cuda":
        setup_cuda_environment()

    # Install llama.cpp
    force_install = args.force_rebuild or args.fast_build or args.scripts_only

    if force_install or not validate_environment():
        if not install_llama_cpp(
            fast_build=args.fast_build, scripts_only=args.scripts_only
        ):
            logger.error("Failed to install llama.cpp")
            sys.exit(1)
    else:
        logger.info("✅ llama.cpp binaries already installed and working")

    # Setup directories (skip if only installing scripts)
    if not args.scripts_only:
        setup_directories()

    # Add installation note
    install_dir = Path.home() / ".local" / "bin"
    if str(install_dir) not in os.environ.get("PATH", ""):
        logger.info(
            f"⚠️  Add {install_dir} to your PATH for global access to llama.cpp binaries"
        )

    logger.info("✅ Setup completed successfully!")
    if args.scripts_only:
        logger.info("📄 Conversion scripts are now available")
    else:
        logger.info(
            "📖 Run 'python setup.py --help' or check README.md for usage instructions"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quantization Utils Setup Script
Initializes environment for bare metal installation with OS detection and llama.cpp setup
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


class SetupLogger:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s]: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("setup.log"),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def info(self, msg):
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)

    def warning(self, msg):
        self.logger.warning(msg)


class EnvironmentDetector:
    def __init__(self, logger):
        self.logger = logger
        self.os_type = platform.system().lower()
        self.arch = platform.machine().lower()
        self.has_nvidia = self._detect_nvidia()
        self.has_metal = self._detect_metal()

    def _detect_nvidia(self):
        """Detect NVIDIA GPU availability"""
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _detect_metal(self):
        """Detect Metal availability (Apple Silicon)"""
        return self.os_type == "darwin" and "arm" in self.arch

    def get_acceleration_type(self):
        """Determine best acceleration type"""
        if self.has_nvidia:
            return "cuda"
        elif self.has_metal:
            return "metal"
        else:
            return "cpu"

    def get_arch_mapping(self):
        """Map platform architecture to llama.cpp naming"""
        arch_map = {
            "x86_64": "x64",
            "amd64": "x64",
            "arm64": "arm64",
            "aarch64": "arm64",
            "armv7l": "armv7l",
        }
        return arch_map.get(self.arch, "x64")

    def get_os_mapping(self):
        """Map OS to llama.cpp naming"""
        if self.os_type == "darwin":
            return "macos"
        elif self.os_type == "linux":
            return "ubuntu"
        elif self.os_type == "windows":
            return "win"
        else:
            raise OSError(f"Unsupported OS: {self.os_type}")


class LlamaCppInstaller:
    def __init__(self, logger, env_detector):
        self.logger = logger
        self.env = env_detector
        self.install_dir = Path.home() / ".local" / "bin"
        self.lib_dir = Path.home() / ".local" / "lib"

    def get_latest_release_info(self):
        """Get latest llama.cpp release information"""
        api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
        try:
            with urllib.request.urlopen(api_url) as response:
                data = json.loads(response.read().decode())
                return data["tag_name"]
        except Exception as e:
            self.logger.error(f"Failed to get latest release: {e}")
            raise

    def download_and_install(self):
        """Download and install appropriate llama.cpp binaries"""
        self.install_dir.mkdir(parents=True, exist_ok=True)
        self.lib_dir.mkdir(parents=True, exist_ok=True)

        latest_version = self.get_latest_release_info()
        arch = self.env.get_arch_mapping()
        os_name = self.env.get_os_mapping()

        # Try different naming conventions
        possible_names = [
            f"llama-{latest_version}-bin-{os_name}-{arch}",
            f"llama-{latest_version}-bin-{os_name}-{self.env.arch}",
            f"llama-{latest_version}-bin-{os_name}",
        ]

        for binary_name in possible_names:
            download_url = f"https://github.com/ggerganov/llama.cpp/releases/download/{latest_version}/{binary_name}.zip"
            self.logger.info(f"Trying to download: {download_url}")

            try:
                zip_path = Path(f"/tmp/{binary_name}.zip")
                urllib.request.urlretrieve(download_url, zip_path)

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    extract_path = Path(f"/tmp/{binary_name}")
                    zip_ref.extractall(extract_path)

                    # Find and copy binaries
                    self._copy_binaries(extract_path)

                zip_path.unlink()  # Clean up
                shutil.rmtree(extract_path, ignore_errors=True)
                self.logger.info("Successfully installed llama.cpp binaries")
                return True

            except Exception as e:
                self.logger.warning(f"Failed to download {binary_name}: {e}")
                continue

        # If all downloads fail, try building from source
        self.logger.warning("Pre-built binaries not available, will build from source")
        return self._build_from_source()

    def _copy_binaries(self, extract_path):
        """Copy binaries to installation directory"""
        bin_dirs = list(extract_path.rglob("bin"))
        if not bin_dirs:
            bin_dirs = [extract_path]

        for bin_dir in bin_dirs:
            if bin_dir.is_dir():
                for binary in bin_dir.iterdir():
                    if binary.is_file() and (
                        binary.suffix in ["", ".exe"] or binary.stat().st_mode & 0o111
                    ):
                        dest = self.install_dir / binary.name
                        shutil.copy2(binary, dest)
                        dest.chmod(0o755)
                        self.logger.info(f"Installed: {dest}")

        # Copy libraries if they exist
        lib_dirs = list(extract_path.rglob("lib"))
        for lib_dir in lib_dirs:
            if lib_dir.is_dir():
                for lib_file in lib_dir.iterdir():
                    if lib_file.is_file() and lib_file.suffix in [
                        ".so",
                        ".dylib",
                        ".dll",
                    ]:
                        dest = self.lib_dir / lib_file.name
                        shutil.copy2(lib_file, dest)
                        self.logger.info(f"Installed library: {dest}")

    def _build_from_source(self):
        """Build llama.cpp from source as fallback"""
        self.logger.info("Building llama.cpp from source...")

        # Clone repository
        repo_dir = Path("/tmp/llama.cpp")
        if repo_dir.exists():
            shutil.rmtree(repo_dir)

        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/ggerganov/llama.cpp.git",
                str(repo_dir),
            ],
            check=True,
        )

        # Configure build
        build_dir = repo_dir / "build"
        build_dir.mkdir(exist_ok=True)

        cmake_args = [
            "cmake",
            "..",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={Path.home() / '.local'}",
        ]

        # Add acceleration-specific flags
        if self.env.has_nvidia:
            cmake_args.extend(["-DGGML_CUDA=ON"])
        elif self.env.has_metal:
            cmake_args.extend(["-DGGML_METAL=ON"])
        else:
            cmake_args.extend(["-DGGML_NATIVE=OFF", "-DGGML_AVX=ON", "-DGGML_AVX2=ON"])

        subprocess.run(cmake_args, cwd=build_dir, check=True)
        subprocess.run(["make", "-j", str(os.cpu_count())], cwd=build_dir, check=True)
        subprocess.run(["make", "install"], cwd=build_dir, check=True)

        # Clean up
        shutil.rmtree(repo_dir)
        self.logger.info("Successfully built and installed llama.cpp from source")
        return True


class DirectorySetup:
    def __init__(self, logger):
        self.logger = logger
        self.base_dir = Path.cwd() / "GGUF"

    def create_directories(self):
        """Create all required directories"""
        directories = ["models", "output", "imatrix", "resources/standard_cal_data"]

        for dir_path in directories:
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {full_path}")

    def download_calibration_data(self):
        """Download standard calibration datasets"""
        cal_data_dir = self.base_dir / "resources" / "standard_cal_data"

        # Sample calibration data (you can extend this)
        datasets = {
            "wiki.utf8": "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/examples/perplexity/wikitext-2-raw/wiki.test.raw",
            "tiny.utf8": "https://gist.githubusercontent.com/karpathy/d4dee566867f8291f086/raw/a348ba29c7e7dd62c6e6d73f22b833e8df55faa5/input.txt",
        }

        for filename, url in datasets.items():
            file_path = cal_data_dir / filename
            if not file_path.exists():
                try:
                    self.logger.info(f"Downloading {filename}...")
                    urllib.request.urlretrieve(url, file_path)
                    self.logger.info(f"Downloaded: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to download {filename}: {e}")


def main():
    logger = SetupLogger()
    logger.info("🚀 Starting Quantization Utils Setup")

    try:
        # Environment detection
        logger.info("🔍 Detecting environment...")
        env = EnvironmentDetector(logger)

        logger.info("📋 Environment Summary:")
        logger.info(f"   OS: {env.os_type}")
        logger.info(f"   Architecture: {env.arch}")
        logger.info(f"   NVIDIA GPU: {env.has_nvidia}")
        logger.info(f"   Metal: {env.has_metal}")
        logger.info(f"   Acceleration: {env.get_acceleration_type()}")

        # Install llama.cpp
        logger.info("📦 Installing llama.cpp...")
        installer = LlamaCppInstaller(logger, env)
        installer.download_and_install()

        # Setup directories and data
        logger.info("📁 Setting up directories...")
        dir_setup = DirectorySetup(logger)
        dir_setup.create_directories()
        dir_setup.download_calibration_data()

        # Update PATH if needed
        local_bin = Path.home() / ".local" / "bin"
        if str(local_bin) not in os.environ.get("PATH", ""):
            logger.info(
                f"⚠️  Add {local_bin} to your PATH for global access to llama.cpp binaries"
            )

        logger.info("✅ Setup completed successfully!")
        logger.info(
            "📖 Run 'python setup.py --help' or check README.md for usage instructions"
        )

    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

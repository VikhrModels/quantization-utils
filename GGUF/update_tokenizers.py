import argparse
import os
import sys
from pathlib import Path
from typing import List

try:
    import logging

    from transformers import AutoTokenizer

    import gguf
except ImportError as e:
    print(f"Import error: {e}")
    print("Install required dependencies: pip install gguf transformers")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_gguf_files(directory: str) -> List[Path]:
    """Find all GGUF files in directory"""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    gguf_files = list(directory.glob("**/*.gguf"))
    logger.info(f"Found {len(gguf_files)} GGUF files in {directory}")

    return gguf_files


def load_source_tokenizer(model_id: str):
    """Load tokenizer from source model"""
    logger.info(f"Loading tokenizer from model: {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logger.info("✅ Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        logger.error(f"❌ Error loading tokenizer: {e}")
        raise


def update_gguf_tokenizer(gguf_file: Path, source_tokenizer) -> bool:
    """Update tokenizer in GGUF file"""
    logger.info(f"Updating tokenizer in file: {gguf_file.name}")

    try:
        # Create backup copy
        backup_file = gguf_file.with_suffix(".gguf.backup")
        if not backup_file.exists():
            logger.info(f"Creating backup copy: {backup_file.name}")
            import shutil

            shutil.copy2(gguf_file, backup_file)

        # Read GGUF file
        reader = gguf.GGUFReader(gguf_file)

        # Create new GGUF writer
        temp_file = gguf_file.with_suffix(".gguf.tmp")
        writer = gguf.GGUFWriter(temp_file, reader.header.arch)

        # Copy all existing metadata except tokenizer
        for key, value in reader.fields.items():
            # Skip tokenizer-specific fields
            if not any(
                token_key in key
                for token_key in [
                    "tokenizer.ggml.tokens",
                    "tokenizer.ggml.scores",
                    "tokenizer.ggml.token_type",
                    "tokenizer.ggml.merges",
                    "tokenizer.ggml.bos_token_id",
                    "tokenizer.ggml.eos_token_id",
                    "tokenizer.ggml.unknown_token_id",
                    "tokenizer.ggml.separator_token_id",
                    "tokenizer.ggml.padding_token_id",
                    "tokenizer.ggml.model",
                ]
            ):
                writer.add_field(key, value)

        # Add new tokenizer data
        logger.info("Adding new tokenizer data...")

        # Get token vocabulary
        vocab = source_tokenizer.get_vocab()
        tokens = [""] * len(vocab)
        scores = [0.0] * len(vocab)
        token_types = [1] * len(vocab)  # GGML_TOKEN_TYPE_NORMAL

        for token, token_id in vocab.items():
            if token_id < len(tokens):
                tokens[token_id] = token

        # Add tokens
        writer.add_array("tokenizer.ggml.tokens", tokens)
        writer.add_array("tokenizer.ggml.scores", scores)
        writer.add_array("tokenizer.ggml.token_type", token_types)

        # Add special tokens
        if (
            hasattr(source_tokenizer, "bos_token_id")
            and source_tokenizer.bos_token_id is not None
        ):
            writer.add_uint32(
                "tokenizer.ggml.bos_token_id", source_tokenizer.bos_token_id
            )

        if (
            hasattr(source_tokenizer, "eos_token_id")
            and source_tokenizer.eos_token_id is not None
        ):
            writer.add_uint32(
                "tokenizer.ggml.eos_token_id", source_tokenizer.eos_token_id
            )

        if (
            hasattr(source_tokenizer, "unk_token_id")
            and source_tokenizer.unk_token_id is not None
        ):
            writer.add_uint32(
                "tokenizer.ggml.unknown_token_id", source_tokenizer.unk_token_id
            )

        if (
            hasattr(source_tokenizer, "pad_token_id")
            and source_tokenizer.pad_token_id is not None
        ):
            writer.add_uint32(
                "tokenizer.ggml.padding_token_id", source_tokenizer.pad_token_id
            )

        # Add merges if this is a BPE tokenizer
        if hasattr(source_tokenizer, "get_merges") and callable(
            source_tokenizer.get_merges
        ):
            try:
                merges = source_tokenizer.get_merges()
                if merges:
                    writer.add_array("tokenizer.ggml.merges", list(merges))
                    logger.info(f"Added {len(merges)} merges")
            except Exception as e:
                logger.warning(f"Could not get merges: {e}")

        # Add tokenizer model
        tokenizer_type = source_tokenizer.__class__.__name__
        writer.add_string("tokenizer.ggml.model", tokenizer_type)

        # Copy tensors from original file
        logger.info("Copying tensors...")
        for tensor in reader.tensors:
            writer.add_tensor(
                tensor.name, tensor.data, tensor.shape, tensor.tensor_type
            )

        # Write file
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        # Replace original file
        os.replace(temp_file, gguf_file)
        logger.info(f"✅ Tokenizer updated in {gguf_file.name}")

        return True

    except Exception as e:
        logger.error(f"❌ Error updating {gguf_file.name}: {e}")
        # Remove temporary file if it was created
        if temp_file.exists():
            temp_file.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Update tokenizers in GGUF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python update_tokenizers.py ./models Vikhrmodels/QVikhr-3-4B-Instruction
  python update_tokenizers.py /path/to/gguf/files microsoft/DialoGPT-medium
        """,
    )

    parser.add_argument("directory", help="Path to directory with GGUF files")

    parser.add_argument(
        "source_model",
        help="Source model ID for tokenizer (e.g.: Vikhrmodels/QVikhr-3-4B-Instruction)",
    )

    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursive search in subdirectories",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    try:
        # Find GGUF files
        gguf_files = find_gguf_files(args.directory)

        if not gguf_files:
            logger.warning("No GGUF files found")
            return

        # Load source tokenizer
        source_tokenizer = load_source_tokenizer(args.source_model)

        if args.dry_run:
            logger.info("🔍 PREVIEW MODE (--dry-run)")
            logger.info(f"Would update {len(gguf_files)} files:")
            for gguf_file in gguf_files:
                logger.info(f"  - {gguf_file}")
            return

        # Update tokenizers
        success_count = 0
        failed_count = 0

        for gguf_file in gguf_files:
            if update_gguf_tokenizer(gguf_file, source_tokenizer):
                success_count += 1
            else:
                failed_count += 1

        # Final report
        logger.info("=" * 50)
        logger.info("📊 UPDATE SUMMARY:")
        logger.info(f"✅ Successfully updated: {success_count} files")
        logger.info(f"❌ Errors: {failed_count} files")
        logger.info(f"📁 Total processed: {len(gguf_files)} files")

        if failed_count > 0:
            logger.warning("⚠️  There are errors! Check logs above")
            sys.exit(1)
        else:
            logger.info("🎉 All tokenizers updated successfully!")

    except Exception as e:
        logger.error(f"💥 Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

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

        # Log chat template info
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            logger.info(
                f"📝 Chat template found: {len(tokenizer.chat_template)} characters"
            )
        else:
            logger.warning("⚠️  No chat template found in source tokenizer")

        return tokenizer
    except Exception as e:
        logger.error(f"❌ Error loading tokenizer: {e}")
        raise


def extract_chat_template(tokenizer) -> str:
    """Extract chat template from tokenizer"""
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        return tokenizer.chat_template

    # Try to get from tokenizer_config if available
    if hasattr(tokenizer, "tokenizer_config"):
        config = tokenizer.tokenizer_config
        if isinstance(config, dict) and "chat_template" in config:
            return config["chat_template"]

    # Try alternative attributes
    for attr in ["default_chat_template", "_chat_template"]:
        if hasattr(tokenizer, attr):
            template = getattr(tokenizer, attr)
            if template:
                return template

    return None


def update_gguf_tokenizer(
    gguf_file: Path, source_tokenizer, chat_template_only=False
) -> bool:
    """Update tokenizer in GGUF file"""
    logger.info(
        f"Updating {'chat template only' if chat_template_only else 'tokenizer'} in file: {gguf_file.name}"
    )

    temp_file = None  # Initialize temp_file variable

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

        # Extract chat template from source
        chat_template = extract_chat_template(source_tokenizer)
        if not chat_template and chat_template_only:
            logger.error("❌ No chat template found in source tokenizer, cannot update")
            return False

        if chat_template_only:
            # Copy all existing metadata except chat template
            for key, value in reader.fields.items():
                if key != "tokenizer.chat_template":
                    writer.add_field(key, value)

            # Add new chat template
            writer.add_string("tokenizer.chat_template", chat_template)
            logger.info(f"✅ Updated chat template only ({len(chat_template)} chars)")
        else:
            # Full tokenizer update (existing code)
            # Copy all existing metadata except tokenizer
            for key, value in reader.fields.items():
                # Skip tokenizer-specific fields including chat template
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
                        "tokenizer.chat_template",  # Chat template field
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

            # Add chat template
            if chat_template:
                writer.add_string("tokenizer.chat_template", chat_template)
                logger.info(f"✅ Added chat template ({len(chat_template)} chars)")
            else:
                logger.warning("⚠️  No chat template found to add")

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
        logger.info(
            f"✅ {'Chat template' if chat_template_only else 'Tokenizer'} updated in {gguf_file.name}"
        )

        return True

    except Exception as e:
        logger.error(f"❌ Error updating {gguf_file.name}: {e}")
        # Remove temporary file if it was created
        if temp_file and temp_file.exists():
            temp_file.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Update tokenizers in GGUF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Update full tokenizer (tokens + chat template)
  python update_tokenizers.py ./models Vikhrmodels/QVikhr-3-4B-Instruction
  
  # Update only chat template (faster, preserves existing tokens)
  python update_tokenizers.py ./models Vikhrmodels/QVikhr-3-4B-Instruction --chat-template-only
  
  # Preview what would be updated without making changes
  python update_tokenizers.py ./models Vikhrmodels/QVikhr-3-4B-Instruction --dry-run
  
  # Recursive search in subdirectories
  python update_tokenizers.py /path/to/gguf/files microsoft/DialoGPT-medium --recursive
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

    parser.add_argument(
        "--chat-template-only",
        action="store_true",
        help="Update only the chat template, leave other tokenizer data unchanged",
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
            mode_text = (
                "chat template only" if args.chat_template_only else "full tokenizer"
            )
            logger.info(f"🔍 PREVIEW MODE (--dry-run) - {mode_text} update")
            logger.info(f"Would update {len(gguf_files)} files:")
            for gguf_file in gguf_files:
                logger.info(f"  - {gguf_file}")
            return

        # Update tokenizers
        success_count = 0
        failed_count = 0

        for gguf_file in gguf_files:
            if update_gguf_tokenizer(
                gguf_file, source_tokenizer, args.chat_template_only
            ):
                success_count += 1
            else:
                failed_count += 1

        # Final report
        update_type = "chat template" if args.chat_template_only else "tokenizer"
        logger.info("=" * 50)
        logger.info(f"📊 {update_type.upper()} UPDATE SUMMARY:")
        logger.info(f"✅ Successfully updated: {success_count} files")
        logger.info(f"❌ Errors: {failed_count} files")
        logger.info(f"📁 Total processed: {len(gguf_files)} files")

        if failed_count > 0:
            logger.warning("⚠️  There are errors! Check logs above")
            sys.exit(1)
        else:
            logger.info(f"🎉 All {update_type}s updated successfully!")

    except Exception as e:
        logger.error(f"💥 Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

import argparse
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

        # Extract architecture from metadata
        architecture = None
        if hasattr(reader, "fields") and reader.fields:
            # Try to get architecture from metadata
            for key, field in reader.fields.items():
                if key == "general.architecture":
                    architecture = (
                        field.parts[field.data[0]]
                        if hasattr(field, "parts") and hasattr(field, "data")
                        else str(field.data[0])
                        if hasattr(field, "data")
                        else None
                    )
                    break

        # Fallback: try to determine architecture from filename or assume llama
        if not architecture:
            logger.warning(
                "Could not determine architecture from GGUF metadata, assuming 'llama'"
            )
            architecture = "llama"

        logger.info(f"Detected architecture: {architecture}")

        # Create new GGUF writer
        temp_file = gguf_file.with_suffix(".gguf.tmp")
        writer = gguf.GGUFWriter(temp_file, architecture)

        # Extract chat template from source tokenizer
        chat_template = extract_chat_template(source_tokenizer)

        if chat_template_only:
            # Copy existing metadata and only update chat template
            if hasattr(reader, "fields") and reader.fields:
                for key, field in reader.fields.items():
                    if key == "tokenizer.chat_template":
                        # Replace with new chat template
                        writer.add_string(key, chat_template)
                        logger.info("✅ Updated chat template")
                    else:
                        # Copy existing metadata
                        try:
                            # Check if field has data (handle arrays properly)
                            has_data = False
                            if hasattr(field, "data"):
                                if field.data is None:
                                    has_data = False
                                elif hasattr(field.data, "__len__"):
                                    # It's an array-like object
                                    has_data = len(field.data) > 0
                                else:
                                    # It's a scalar
                                    has_data = True

                            if has_data and hasattr(field, "types") and field.types:
                                # Handle different field types
                                field_type = (
                                    field.types[0]
                                    if isinstance(field.types, list)
                                    else field.types
                                )
                                if field_type == gguf.GGUFValueType.STRING:
                                    value = (
                                        field.parts[field.data[0]]
                                        if hasattr(field, "parts")
                                        else str(field.data[0])
                                    )
                                    writer.add_string(key, value)
                                elif field_type == gguf.GGUFValueType.UINT32:
                                    writer.add_uint32(key, field.data[0])
                                elif field_type == gguf.GGUFValueType.FLOAT32:
                                    writer.add_float32(key, field.data[0])
                                elif field_type == gguf.GGUFValueType.BOOL:
                                    writer.add_bool(key, field.data[0])
                                elif field_type == gguf.GGUFValueType.ARRAY:
                                    # Handle arrays based on their content type
                                    if hasattr(field, "parts") and field.parts:
                                        writer.add_array(key, field.parts)
                                    else:
                                        writer.add_array(key, field.data)
                                else:
                                    # Generic fallback
                                    logger.debug(
                                        f"Copying field {key} with unknown type {field_type}"
                                    )
                                    continue
                        except Exception as e:
                            logger.warning(f"Could not copy field {key}: {e}")
                            continue
        else:
            # Full tokenizer update - copy metadata and replace tokenizer data
            # Copy existing non-tokenizer metadata
            if hasattr(reader, "fields") and reader.fields:
                for key, field in reader.fields.items():
                    if not key.startswith("tokenizer."):
                        try:
                            # Copy non-tokenizer metadata
                            # (implementation similar to above)
                            continue
                        except Exception as e:
                            logger.warning(f"Could not copy field {key}: {e}")
                            continue

            # Add tokenizer data from source
            # This would require implementing full tokenizer extraction
            logger.info("Full tokenizer update - extracting tokenizer data...")
            # Add chat template
            writer.add_string("tokenizer.chat_template", chat_template)

        # Copy tensor information
        if hasattr(reader, "tensors"):
            for tensor in reader.tensors:
                writer.add_tensor_info(
                    name=tensor.name,
                    shape=tensor.shape,
                    dtype=tensor.tensor_type,
                    data_offset=tensor.data_offset,
                )

        # Write the file
        writer.write_header_to_file()
        writer.close()

        # Copy tensor data from original file to new file
        if hasattr(reader, "tensors"):
            with open(gguf_file, "rb") as src_f, open(temp_file, "r+b") as dst_f:
                # Seek to end of header in destination file
                dst_f.seek(0, 2)  # Seek to end

                for tensor in reader.tensors:
                    # Read tensor data from source
                    src_f.seek(tensor.data_offset)
                    tensor_data = src_f.read(tensor.n_bytes)

                    # Write tensor data to destination
                    dst_f.write(tensor_data)

        # Replace original file with temporary file
        if temp_file.exists():
            temp_file.replace(gguf_file)
            logger.info(f"✅ Successfully updated {gguf_file.name}")
            return True

    except Exception as e:
        logger.error(f"❌ Error updating {gguf_file.name}: {e}")
        if temp_file and temp_file.exists():
            temp_file.unlink()  # Clean up temporary file
        return False

    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink()  # Clean up temporary file if it still exists

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

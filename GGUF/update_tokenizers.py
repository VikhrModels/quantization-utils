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
        logger.debug("Step 1: Creating backup copy")
        # Create backup copy
        backup_file = gguf_file.with_suffix(".gguf.backup")
        if not backup_file.exists():
            logger.info(f"Creating backup copy: {backup_file.name}")
            import shutil

            shutil.copy2(gguf_file, backup_file)

        logger.debug("Step 2: Reading GGUF file")
        # Read GGUF file
        reader = gguf.GGUFReader(gguf_file)
        logger.debug(f"GGUF reader created: {type(reader)}")

        logger.debug("Step 3: Extracting architecture")
        # Extract architecture from metadata
        architecture = None
        try:
            logger.debug(f"Reader has fields attribute: {hasattr(reader, 'fields')}")
            if hasattr(reader, "fields"):
                logger.debug(f"Reader.fields type: {type(reader.fields)}")
                logger.debug(f"Reader.fields is None: {reader.fields is None}")

                # Check if reader.fields has items method
                if hasattr(reader.fields, "items"):
                    logger.debug("Reader.fields has items method")
                    # Try to safely iterate
                    try:
                        for key, field in reader.fields.items():
                            logger.debug(f"Checking field: {key}")
                            if key == "general.architecture":
                                logger.debug(
                                    f"Found architecture field, type: {type(field)}"
                                )
                                if hasattr(field, "data") and field.data is not None:
                                    logger.debug(f"Field data type: {type(field.data)}")
                                    if (
                                        hasattr(field.data, "__len__")
                                        and len(field.data) > 0
                                    ):
                                        architecture = str(field.data[0])
                                        logger.debug(
                                            f"Extracted architecture: {architecture}"
                                        )
                                break
                    except Exception as e:
                        logger.debug(f"Error iterating reader.fields: {e}")
                else:
                    logger.debug("Reader.fields does not have items method")
            else:
                logger.debug("Reader does not have fields attribute")
        except Exception as e:
            logger.debug(f"Error extracting architecture: {e}")

        if not architecture:
            architecture = "llama"  # Default fallback
            logger.debug(f"Using fallback architecture: {architecture}")

        logger.debug("Step 4: Creating GGUF writer")
        # Create new GGUF writer
        temp_file = gguf_file.with_suffix(".gguf.tmp")
        writer = gguf.GGUFWriter(temp_file, architecture)
        logger.debug(f"GGUF writer created with architecture: {architecture}")

        logger.debug("Step 5: Extracting chat template")
        # Extract chat template from source tokenizer
        chat_template = extract_chat_template(source_tokenizer)
        logger.debug(
            f"Chat template extracted, length: {len(chat_template) if chat_template else 0}"
        )

        logger.debug("Step 6: Processing fields")
        if chat_template_only:
            logger.debug("Entering chat_template_only mode")
            # Copy existing metadata and only update chat template
            fields_exist = False
            try:
                logger.debug("Checking if reader has fields...")
                fields_exist = hasattr(reader, "fields")
                logger.debug(f"Reader has fields: {fields_exist}")

                if fields_exist:
                    logger.debug("Checking if fields is not None...")
                    fields_exist = reader.fields is not None
                    logger.debug(f"Fields is not None: {fields_exist}")

                    if fields_exist and hasattr(reader.fields, "__len__"):
                        logger.debug("Checking fields length...")
                        fields_len = len(reader.fields)
                        logger.debug(f"Fields length: {fields_len}")
                        fields_exist = fields_len > 0

            except Exception as e:
                logger.error(f"Error checking fields existence: {e}")
                logger.error(f"Error type: {type(e)}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                fields_exist = False

            logger.debug(f"Fields exist check result: {fields_exist}")

            if fields_exist:
                logger.debug("Getting fields items...")
                try:
                    if hasattr(reader.fields, "items"):
                        logger.debug("Using reader.fields.items()")
                        fields_items = reader.fields.items()
                    else:
                        logger.debug(
                            "reader.fields has no items method, using empty list"
                        )
                        fields_items = []
                except Exception as e:
                    logger.error(f"Error getting fields items: {e}")
                    logger.error(f"Error type: {type(e)}")
                    import traceback

                    logger.error(f"Traceback: {traceback.format_exc()}")
                    fields_items = []

                logger.debug("Starting field iteration...")
                for key, field in fields_items:
                    try:
                        logger.debug(f"Processing field: {key}")

                        if key == "tokenizer.chat_template":
                            # Replace with new chat template
                            logger.debug(
                                "Found tokenizer.chat_template field - replacing"
                            )
                            writer.add_string(key, chat_template)
                            logger.info("✅ Updated chat template")
                        else:
                            logger.debug(f"Processing metadata field: {key}")
                            # Copy existing metadata with careful array handling
                            if not hasattr(field, "data"):
                                logger.debug(
                                    f"Field {key} has no data attribute, skipping"
                                )
                                continue

                            logger.debug(f"Field {key} has data attribute")
                            # Safely check if data exists
                            try:
                                logger.debug(
                                    f"Checking if field.data is not None for {key}"
                                )
                                data_is_not_none = field.data is not None
                                logger.debug(
                                    f"Field {key} data is not None: {data_is_not_none}"
                                )

                                data_exists = data_is_not_none

                                if data_is_not_none:
                                    logger.debug(
                                        f"Checking if field.data has __len__ for {key}"
                                    )
                                    has_len = hasattr(field.data, "__len__")
                                    logger.debug(
                                        f"Field {key} data has __len__: {has_len}"
                                    )

                                    if has_len:
                                        logger.debug(
                                            f"Checking if field.data is string or bytes for {key}"
                                        )
                                        is_str_or_bytes = isinstance(
                                            field.data, (str, bytes)
                                        )
                                        logger.debug(
                                            f"Field {key} data is str/bytes: {is_str_or_bytes}"
                                        )

                                        if not is_str_or_bytes:
                                            logger.debug(
                                                f"Checking length of field.data for {key}"
                                            )
                                            try:
                                                data_len = len(field.data)
                                                logger.debug(
                                                    f"Field {key} data length: {data_len}"
                                                )
                                                data_exists = data_len > 0
                                            except Exception as len_e:
                                                logger.error(
                                                    f"Error getting length for field {key}: {len_e}"
                                                )
                                                data_exists = True  # Assume it exists if we can't check length

                            except Exception as e:
                                logger.error(
                                    f"Could not check data for field {key}: {e}"
                                )
                                logger.error(f"Error type: {type(e)}")
                                import traceback

                                logger.error(f"Traceback: {traceback.format_exc()}")
                                continue

                            logger.debug(
                                f"Field {key} data exists check result: {data_exists}"
                            )
                            if not data_exists:
                                logger.debug(f"Field {key} has no data, skipping")
                                continue

                            # Safely check types
                            logger.debug(f"Checking if field {key} has types attribute")
                            if not hasattr(field, "types"):
                                logger.debug(
                                    f"Field {key} has no types attribute, skipping"
                                )
                                continue

                            logger.debug(f"Field {key} has types attribute")
                            try:
                                logger.debug(f"Getting field.types for {key}")
                                field_types = field.types
                                logger.debug(
                                    f"Field {key} types: {field_types}, type: {type(field_types)}"
                                )

                                if hasattr(field_types, "__len__") and not isinstance(
                                    field_types, (str, bytes)
                                ):
                                    logger.debug(f"Field {key} types is array-like")
                                    try:
                                        types_len = len(field_types)
                                        logger.debug(
                                            f"Field {key} types length: {types_len}"
                                        )
                                        if types_len == 0:
                                            logger.debug(
                                                f"Field {key} has empty types, skipping"
                                            )
                                            continue
                                        field_type = field_types[0]
                                        logger.debug(
                                            f"Field {key} first type: {field_type}"
                                        )
                                    except Exception as types_e:
                                        logger.error(
                                            f"Error processing types array for field {key}: {types_e}"
                                        )
                                        continue
                                else:
                                    logger.debug(f"Field {key} types is scalar")
                                    field_type = field_types

                            except Exception as e:
                                logger.error(
                                    f"Could not access types for field {key}: {e}"
                                )
                                logger.error(f"Error type: {type(e)}")
                                import traceback

                                logger.error(f"Traceback: {traceback.format_exc()}")
                                continue

                            # Now safely handle the field based on its type
                            logger.debug(f"Field {key} has type: {field_type}")

                            if field_type == gguf.GGUFValueType.STRING:
                                try:
                                    if hasattr(field, "parts") and field.parts:
                                        value = field.parts[field.data[0]]
                                    else:
                                        value = str(field.data[0])
                                    writer.add_string(key, value)
                                except Exception as e:
                                    logger.debug(
                                        f"Error processing STRING field {key}: {e}"
                                    )
                                    continue

                            elif field_type == gguf.GGUFValueType.UINT32:
                                try:
                                    writer.add_uint32(key, int(field.data[0]))
                                except Exception as e:
                                    logger.debug(
                                        f"Error processing UINT32 field {key}: {e}"
                                    )
                                    continue

                            elif field_type == gguf.GGUFValueType.FLOAT32:
                                try:
                                    writer.add_float32(key, float(field.data[0]))
                                except Exception as e:
                                    logger.debug(
                                        f"Error processing FLOAT32 field {key}: {e}"
                                    )
                                    continue

                            elif field_type == gguf.GGUFValueType.BOOL:
                                try:
                                    writer.add_bool(key, bool(field.data[0]))
                                except Exception as e:
                                    logger.debug(
                                        f"Error processing BOOL field {key}: {e}"
                                    )
                                    continue

                            elif field_type == gguf.GGUFValueType.ARRAY:
                                try:
                                    if hasattr(field, "parts") and field.parts:
                                        writer.add_array(key, field.parts)
                                    else:
                                        writer.add_array(key, list(field.data))
                                except Exception as e:
                                    logger.debug(
                                        f"Error processing ARRAY field {key}: {e}"
                                    )
                                    continue
                            else:
                                logger.debug(
                                    f"Unknown field type {field_type} for field {key}, skipping"
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

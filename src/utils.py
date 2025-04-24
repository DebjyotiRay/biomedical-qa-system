"""
Utility functions for the Biomedical QA System
"""

import os
import json
import logging
from typing import List, Dict, Any, Set, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv # Import dotenv

# Import constants needed
from src.constants import QUERY_TYPE_INDICATORS

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("biomedical_qa.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_environment_variables():
    """Loads environment variables from .env file."""
    loaded = load_dotenv()
    if loaded:
        logger.info(".env file loaded successfully.")
    else:
        logger.warning(".env file not found or empty. Relying on system environment variables.")
    return loaded

def load_config(config_path: str) -> Dict[str, Any]:
    """ Load configuration from JSON file with error handling """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found. Exiting.")
        raise # Re-raise exception
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing config file {config_path}: {e}. Exiting.")
        raise # Re-raise exception

# Entity extraction moved to citations.py

def determine_query_type(query: str) -> str:
    """ Determine the type of biomedical query """
    # Import here to avoid potential circular dependency if moved back later
    from src.citations import extract_entities_from_text

    query_lower = query.lower()

    # Check for specific indicators in the query
    for query_type, indicators in QUERY_TYPE_INDICATORS.items():
        if any(indicator in query_lower for indicator in indicators):
            return query_type

    # If no clear indicators, use entity extraction as a fallback
    entities = extract_entities_from_text(query)

    # Prioritize based on entity presence
    if entities.get("genes"): return "gene"
    if entities.get("proteins"): return "protein"
    if entities.get("diseases"): return "disease"
    if entities.get("pathways"): return "pathway"

    return "general"

def format_scientific_answer(answer: str) -> str:
    """ Format scientific answers for consistency, especially for bullet points """
    lines = answer.strip().split("\n")
    formatted_lines = []
    has_bullet_points = any(line.strip().startswith(("•", "-", "*")) for line in lines)

    if has_bullet_points:
        for line in lines:
            line = line.strip()
            if not line: continue
            if line.startswith(("•", "-", "*")):
                entity = line[1:].strip()
                # Avoid adding empty bullets
                if entity:
                     formatted_lines.append(f"• {entity}")
            # Ignore lines that don't start with a bullet in a bulleted list context
    else:
        # Try to convert comma/semicolon separated lists
        items = []
        # Prioritize semicolon if present, as commas might be within items
        if ";" in answer and len(answer.split(";")) > 1: # Check > 1 for single items with semicolons
             items = [item.strip() for item in answer.split(";") if item.strip()]
        elif "," in answer and len(answer.split(",")) > 1: # Check > 1 for single items with commas
             items = [item.strip() for item in answer.split(",") if item.strip()]

        if items:
            formatted_lines = [f"• {item}" for item in items]
        else:
            # If not list-like, return the original (stripped) answer
            # Avoid returning single-line answers as bullet points unless it's the only line
            if len(lines) == 1:
                return lines[0]
            else: # Keep multi-line non-bullet answers as they are
                formatted_lines = [line.strip() for line in lines if line.strip()]


    return "\n".join(formatted_lines)

def evaluate_scientific_answers(generated_answers: List[str], reference_answers: List[str]) -> Dict[str, float]:
    """ Evaluate scientific answer precision by comparing entity sets """
    def extract_entities(answer):
        entities = set()
        # Normalize potential variations in bullet points or list items
        for line in answer.split("\n"):
            line = line.strip()
            if line.startswith(("•", "-", "*")):
                entity = line[1:].strip().lower() # Compare lower case
                if entity: entities.add(entity)
            # Consider non-bullet lines as potential single entities if list is short? Risky.
            # Let's stick to only extracting from explicit list markers for precision.
        return entities

    total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0
    count = 0

    for gen, ref in zip(generated_answers, reference_answers):
        gen_entities = extract_entities(gen)
        ref_entities = extract_entities(ref)

        # Skip evaluation if reference is empty
        if not ref_entities:
            continue # Skip pairs where reference is empty

        count += 1 # Only count pairs with non-empty references

        if not gen_entities: # Generated is empty, but reference is not
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            true_positives = len(gen_entities.intersection(ref_entities))
            precision = true_positives / len(gen_entities)
            recall = true_positives / len(ref_entities) # ref_entities is non-empty here
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    num_examples = count if count > 0 else 1 # Avoid division by zero
    return {
        "precision": total_precision / num_examples,
        "recall": total_recall / num_examples,
        "f1": total_f1 / num_examples
    }

def export_results(answer: str, format: str = "txt") -> Dict[str, Any]:
    """ Export the results in various formats """
    export_data = {"content": answer, "format": format}
    entities = []

    # Extract bullet points into structured data
    for line in answer.split("\n"):
        line = line.strip()
        if line.startswith(("•", "-", "*")):
            entity = line[1:].strip()
            if entity: entities.append(entity)

    if entities: # Only add structured content if entities were extracted
        if format == "csv":
            # Simple CSV: one column named 'entity'
            # Ensure proper CSV quoting if entities contain commas or quotes
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output, quoting=csv.QUOTE_ALL)
            writer.writerow(['entity']) # Header
            for entity in entities:
                 writer.writerow([entity])
            export_data["structured_content"] = output.getvalue()

        elif format == "json":
            json_content = {"entities": entities}
            export_data["structured_content"] = json.dumps(json_content, indent=2)
        elif format == "txt": # For txt, structured might just be the list
             export_data["structured_content"] = "\n".join(entities)

    # Add timestamp
    export_data["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    return export_data

def save_export(export_data: Dict[str, Any], filename: Optional[str] = None) -> str:
    """ Save exported data to a file """
    output_dir = export_data.get("output_dir", ".") # Get output dir if specified
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        timestamp = export_data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        filename = f"biomedical_answer_{timestamp}.{export_data['format']}"

    full_path = os.path.join(output_dir, filename)

    try:
        content_to_save = export_data.get("structured_content", export_data.get("content", ""))
        with open(full_path, 'w', encoding='utf-8') as f: # Specify encoding
            f.write(content_to_save)
        logger.info(f"Results exported to {full_path}")
        return full_path
    except Exception as e:
        logger.error(f"Error exporting results to {full_path}: {e}")
        return ""

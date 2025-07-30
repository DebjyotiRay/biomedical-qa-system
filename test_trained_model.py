#!/usr/bin/env python3
"""
Test script for the fine-tuned biomedical QA model
"""

import logging
from src.model import BiomedicalModel
from src.utils import load_config, load_environment_variables

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    """Test the fine-tuned model with sample questions"""
    
    # Load configuration
    load_environment_variables()
    config = load_config("config.json")
    
    # Initialize model
    logger.info("Loading model...")
    model = BiomedicalModel(config)
    
    # Load base model
    if not model.load_model():
        logger.error("Failed to load base model")
        return False
    
    # Load the fine-tuned adapter
    adapter_path = "output/peft_adapter"
    if model.load_peft_adapter(adapter_path):
        logger.info("Fine-tuned adapter loaded successfully!")
    else:
        logger.warning("No fine-tuned adapter found. Using base model.")
    
    # Test questions
    test_questions = [
        "What genes are associated with Parkinson's disease?",
        "Which proteins are involved in DNA repair?",
        "What drugs target EGFR?",
        "Which genes are mutated in breast cancer?",
        "What proteins regulate autophagy?",
        "Which drugs are used to treat multiple sclerosis?",
        "What genes are involved in Alzheimer's disease?",
        "Which proteins are part of the PI3K/AKT pathway?"
    ]
    
    print("\n" + "="*60)
    print("TESTING FINE-TUNED BIOMEDICAL QA MODEL")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 50)
        
        try:
            answer = model.generate_answer(question)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error generating answer: {e}")
        
        print("-" * 50)
    
    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_model()

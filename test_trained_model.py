#!/usr/bin/env python3
"""
Test script for the fine-tuned biomedical QA model
"""

import logging
from src.model import BiomedicalModel
from src.qa_handler import BiomedicalQAHandler
from src.utils import load_config, load_environment_variables

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    """Test the fine-tuned model with sample questions"""
    
    # Load configuration
    load_environment_variables()
    config = load_config("config.json")
    
    # Check if direct fallback is enabled
    use_direct_fallback = config.get("fallback", {}).get("use_direct_fallback", False)
    
    if use_direct_fallback:
        logger.info("Direct fallback is enabled - testing with fallback pipeline")
        return test_with_fallback(config)
    else:
        logger.info("Testing with trained model")
        return test_with_trained_model(config)

def test_with_trained_model(config):
    """Test using the trained model directly"""
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
    
    return run_test_questions_with_model(model)

def test_with_fallback(config):
    """Test using the fallback pipeline"""
    logger.info("Setting up QA handler with fallback pipeline...")
    
    # Initialize QA handler
    qa_handler = BiomedicalQAHandler(config)
    
    # Set up fallback clients
    qa_handler._setup_fallback_clients()
    
    # Override setup to bypass model loading (similar to test_direct_fallback_live.py)
    def bypass_setup(self):
        logger.info("Bypassing model loading for direct fallback test")
        return True
    
    # Replace the original setup method with our bypass
    qa_handler.setup = bypass_setup.__get__(qa_handler, BiomedicalQAHandler)
    
    # Call our bypassed setup method
    if not qa_handler.setup():
        logger.error("Failed to set up QA system")
        return False
    
    return run_test_questions_with_handler(qa_handler)

def run_test_questions_with_model(model):
    """Run test questions using the model directly"""
    
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

def run_test_questions_with_handler(qa_handler):
    """Run test questions using the QA handler with fallback"""
    
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
    print("TESTING BIOMEDICAL QA WITH DIRECT FALLBACK")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 50)
        
        try:
            result = qa_handler.answer_question(question, include_citations=True)
            print(f"Answer: {result['formatted_answer']}")
            print(f"Source: {result['answer_source']}")
            if result.get('citations'):
                print(f"Citations: {len(result['citations'])} found")
        except Exception as e:
            print(f"Error generating answer: {e}")
            logger.error(f"Error processing question '{question}': {e}", exc_info=True)
        
        print("-" * 50)
    
    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_model()

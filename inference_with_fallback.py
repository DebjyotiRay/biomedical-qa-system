#!/usr/bin/env python3
"""
Inference script for the biomedical QA system that loads the trained model
but uses the direct fallback pipeline for answer generation
"""

import os
import sys
import json
import logging
import time

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.qa_handler import BiomedicalQAHandler
from src.utils import load_config, load_environment_variables

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/inference_fallback.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FallbackInferenceHandler(BiomedicalQAHandler):
    """
    Extended QA handler that loads the model but uses direct fallback for inference
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.model_loaded = False
        
    def setup_with_fallback_inference(self):
        """
        Set up the system: load the model but configure for fallback inference
        """
        logger.info("Setting up inference system with fallback pipeline...")
        
        # Check if direct fallback is enabled in config
        use_direct_fallback = self.config.get("fallback", {}).get("use_direct_fallback", False)
        
        if use_direct_fallback:
            logger.info("Direct fallback enabled - model will be loaded but inference will use fallback")
            
            # Load the model (for completeness and potential future use)
            logger.info("Loading trained model...")
            if self.model.load_model():
                logger.info("Base model loaded successfully")
                
                # Try to load the fine-tuned adapter
                adapter_path = "output/peft_adapter"
                if self.model.load_peft_adapter(adapter_path):
                    logger.info("Fine-tuned adapter loaded successfully!")
                    self.model_loaded = True
                else:
                    logger.warning("No fine-tuned adapter found. Base model loaded.")
                    self.model_loaded = True
            else:
                logger.warning("Failed to load base model, proceeding with fallback only")
            
            # Set up fallback clients (this is the key part)
            self._setup_fallback_clients()
            
            # Override the inference configuration to ensure fallback is used
            if "inference" not in self.config:
                self.config["inference"] = {}
            self.config["inference"]["use_llm_parent"] = False
            
            logger.info("System configured for direct fallback inference")
            return True
        else:
            logger.info("Direct fallback not enabled, using standard setup")
            return self.setup()
    
    def generate_answer_with_fallback(self, question: str, include_citations: bool = True):
        """
        Generate answer using the fallback pipeline while having the model loaded
        """
        logger.info(f"Generating answer using fallback pipeline for: {question}")
        
        # Use the parent class method which will automatically use fallback
        # since use_llm_parent is set to False
        result = self.answer_question(question, include_citations)
        
        # Add information about the loaded model for reference
        if self.model_loaded:
            result["model_status"] = "loaded_but_using_fallback"
            result["adapter_loaded"] = self.model.peft_model is not None
        else:
            result["model_status"] = "not_loaded_using_fallback_only"
            result["adapter_loaded"] = False
            
        return result

def test_inference_with_fallback():
    """Test the inference system with fallback pipeline"""
    
    # Load environment variables and config
    load_environment_variables()
    config = load_config("config.json")
    
    # Initialize the fallback inference handler
    handler = FallbackInferenceHandler(config)
    
    # Set up the system
    if not handler.setup_with_fallback_inference():
        logger.error("Failed to set up inference system")
        return False
    
    # Test questions
    test_questions = [
        "What genes are associated with Parkinson's disease?",
        "Which proteins are involved in DNA repair?",
        "What drugs target EGFR?",
        "Which genes are mutated in breast cancer?",
        "How many days will it take to aggregate parkinsons disease pathology in primary neurons?"
    ]
    
    print("\n" + "="*80)
    print("BIOMEDICAL QA INFERENCE WITH FALLBACK PIPELINE")
    print("="*80)
    print("Model Status: Loaded but using direct fallback for inference")
    print("="*80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 60)
        
        try:
            start_time = time.time()
            result = handler.generate_answer_with_fallback(question, include_citations=True)
            elapsed_time = time.time() - start_time
            
            print(f"Answer: {result['formatted_answer']}")
            print(f"\nSource: {result['answer_source']}")
            print(f"Model Status: {result['model_status']}")
            print(f"Adapter Loaded: {result['adapter_loaded']}")
            print(f"Processing Time: {elapsed_time:.2f}s")
            
            if result.get('citations'):
                print(f"Citations Found: {len(result['citations'])}")
            
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            print(f"Error: {e}")
        
        print("-" * 60)
    
    print("\n" + "="*80)
    print("INFERENCE TESTING COMPLETED")
    print("="*80)
    
    return True

def interactive_inference():
    """Run interactive inference session with fallback pipeline"""
    
    # Load environment variables and config
    load_environment_variables()
    config = load_config("config.json")
    
    # Initialize the fallback inference handler
    handler = FallbackInferenceHandler(config)
    
    # Set up the system
    if not handler.setup_with_fallback_inference():
        logger.error("Failed to set up inference system")
        return False
    
    print("\n" + "="*80)
    print("INTERACTIVE BIOMEDICAL QA WITH FALLBACK INFERENCE")
    print("="*80)
    print("Model: Loaded but using direct fallback pipeline")
    print("Type 'exit' to quit, 'test' to run test questions")
    print("="*80)
    
    while True:
        try:
            user_input = input("\nEnter your biomedical question: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            elif user_input.lower() == 'test':
                test_inference_with_fallback()
                continue
            elif not user_input:
                continue
            
            print("\nProcessing...")
            start_time = time.time()
            result = handler.generate_answer_with_fallback(user_input, include_citations=True)
            elapsed_time = time.time() - start_time
            
            print("\n" + "="*60)
            print(result["formatted_answer"])
            print("="*60)
            print(f"Source: {result['answer_source']} | Model: {result['model_status']} | Time: {elapsed_time:.2f}s")
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Type 'exit' to quit.")
            break
        except Exception as e:
            logger.error(f"Error during interactive session: {e}", exc_info=True)
            print(f"Error: {e}")
    
    print("\nSession ended.")
    return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Biomedical QA Inference with Fallback Pipeline")
    parser.add_argument("--mode", choices=["test", "interactive"], default="test",
                       help="Run mode: test (run test questions) or interactive (interactive session)")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        success = test_inference_with_fallback()
    else:
        success = interactive_inference()
    
    if success:
        logger.info("Inference session completed successfully")
    else:
        logger.error("Inference session failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple entry point for the Biomedical QA System
"""
import os
import sys
import argparse
from src.qa_handler import BiomedicalQAHandler
from src.utils import load_config, load_environment_variables

def main():
    """Main function to start the Biomedical QA system"""
    parser = argparse.ArgumentParser(description="Biomedical QA System")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--question", help="Process a single question and exit")
    parser.add_argument("--no-citations", action="store_true", help="Disable citation fetching")
    args = parser.parse_args()
    
    # Load environment variables and configuration
    load_environment_variables()
    config = load_config("config.json")
    
    # Initialize QA handler
    qa_handler = BiomedicalQAHandler(config)
    
    # Process a single question or run in interactive mode
    if args.question:
        result = qa_handler.answer_question(args.question, include_citations=not args.no_citations)
        print("\n" + "="*80)
        print(result["formatted_answer"])
        print("="*80)
        print(f"(Source: {result['answer_source']} | Time: {result['processing_time_seconds']}s)")
    else:
        # Default to interactive mode
        qa_handler.interactive_session()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

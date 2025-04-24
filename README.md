# Biomedical QA System with PubMed Citation Integration

A biomedical question answering system that provides answers with relevant scientific citations from PubMed.

## Features

- Answer biomedical questions with LLM-powered responses
- Automatically retrieve relevant PubMed citations to support answers
- Entity-based query optimization for improved citation relevance
- Fallback to OpenAI GPT models when needed

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API keys in a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key
NCBI_API_KEY=your_ncbi_api_key
```

3. Run in interactive mode:
```bash
python main.py --interactive
```

4. Or ask a single question:
```bash
python main.py --question "What genes are associated with Parkinson's disease?"
```

## System Components

- **QA Handler**: Coordinates the question answering process
- **Citations Module**: Retrieves relevant scientific publications from PubMed
- **Model Module**: Manages language model interaction and fallback mechanisms
- **Entity Extraction**: Identifies biomedical entities for better search results

## Documentation

See the `docs` directory for:
- System architecture diagram
- Training guide for fine-tuning models

## Testing

Run the test script to verify the system is working properly:
```bash
python biomed_qa_system_final/test_direct_fallback_live.py
```

## Configuration

The system can be customized through the `config.json` file, including:
- PubMed citation parameters
- Model selection and configuration
- Fallback model settings

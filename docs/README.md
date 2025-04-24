# Biomedical QA System

A sophisticated question-answering system for biomedical research that combines state-of-the-art language models with real-time PubMed citations.

## Key Features

- Answer complex biomedical questions with structured, scientifically accurate responses
- Automatically fetch relevant, recent PubMed citations to support answers
- Generate domain-specific answers for genes, proteins, diseases, and biological pathways
- Provide fallback mechanisms using OpenAI, NVIDIA, or Google models when needed
- Support for Parameter-Efficient Fine-Tuning (PEFT) with LoRA to specialize the model
- Configurable proxy settings for enterprise environments
- Interactive, batch processing, and API modes

## System Architecture

The system consists of several integrated components:

1. **QA Handler**: Orchestrates the question-answering process
2. **Biomedical Model**: Handles inference with support for fine-tuning
3. **Citations Module**: Provides real-time scientific citations from PubMed
4. **Utilities**: Support functions for evaluation, formatting, and configuration

![System Architecture](https://github.com/username/biomedical-qa-system/raw/main/docs/images/system_architecture.png)

For detailed architecture diagrams, see [architecture_diagram.md](architecture_diagram.md).

## Installation

```bash
# Clone the repository
git clone https://github.com/username/biomedical-qa-system.git
cd biomedical-qa-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_key
NCBI_API_KEY=your_ncbi_key
GOOGLE_API_KEY=your_google_key  # Optional
NVIDIA_API_KEY=your_nvidia_key  # Optional
```

Customize the system behavior by editing `config.json`.

## Quick Start

### Interactive Mode

```bash
python run.py --interactive
```

### API Mode

```bash
python run.py --api --port 8000
```

Then send requests to:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What genes are associated with Parkinson disease?"}'
```

### Batch Mode

```bash
python run.py --batch questions.txt --output results.json
```

## Sample Usage

```python
from qa_handler import BiomedicalQAHandler
from utils import load_config, load_environment_variables

# Load configuration
load_environment_variables()
config = load_config("config.json")

# Initialize QA handler
qa_handler = BiomedicalQAHandler(config)
qa_handler.setup()

# Ask a question with citations
result = qa_handler.answer_question(
    "What genes are associated with Parkinson's disease?",
    include_citations=True
)

# Print the result
print(result["formatted_answer"])
```

## Training and Fine-tuning

The system supports fine-tuning the biomedical model using LoRA (Low-Rank Adaptation), allowing efficient customization with domain-specific data.

For detailed fine-tuning instructions, see [training_guide.md](training_guide.md).

## Citation Integration

Our system provides real-time, high-quality citations from PubMed to support its answers. The citation system:

1. Extracts biomedical entities from the question
2. Generates optimized PubMed search queries
3. Retrieves and ranks citations based on relevance, recency, and article type
4. Presents citations in a structured format with links to the source papers

The citation retrieval can be configured to prioritize review articles, control the maximum age of citations, and customize network settings including proxies.

## Performance Evaluation

The system includes tools to evaluate:

- Answer accuracy using precision, recall, and F1 score
- Citation quality metrics
- Processing time and fallback usage
- Fine-tuning effectiveness

## Advanced Usage

- **Entity Visualization**: Generate network visualizations of genes, proteins, or pathways
- **Multi-step Queries**: Chain queries for complex research questions
- **Export Results**: Save answers in various formats (JSON, CSV, TXT)
- **Custom Models**: Configure different base models or fallback options

## Documentation

- [Architecture Diagram](architecture_diagram.md) - Detailed system architecture
- [Training Guide](training_guide.md) - Instructions for fine-tuning the model
- [Configuration Reference](docs/configuration.md) - Configuration options
- [API Reference](docs/api.md) - API endpoints and usage examples

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT 0.4+
- BitsAndBytes (optional, for 4-bit quantization)
- FastAPI, uvicorn (optional, for API mode)

## Citation

If you use this system in your research, please cite:

```
@software{biomedical_qa_system,
  author = {DebjyotiRay},
  title = {Biomedical QA System},
  year = {2025},
  url = {https://github.com/DebjyotiRay/biomedical-qa-system}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

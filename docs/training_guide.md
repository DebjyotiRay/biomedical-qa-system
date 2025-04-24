# Biomedical QA System: Training & Usage Guide

This document provides detailed instructions for training, finetuning, and using the biomedical QA system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Setup & Installation](#setup--installation)
3. [Configuration Options](#configuration-options)
4. [Training Process](#training-process)
5. [Running the System](#running-the-system)
6. [Evaluating Performance](#evaluating-performance)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware Requirements

For training:
- GPU with at least 8GB VRAM (16GB+ recommended for larger models)
- 16GB+ system RAM
- 100GB+ storage space for models and data

For inference:
- GPU with 8GB+ VRAM for optimal performance
- CPU-only mode available but will be significantly slower
- 8GB+ system RAM
- 20GB+ storage space for models

### Software Requirements

- Python 3.8+ 
- PyTorch 2.0+
- Transformers library 4.30+
- PEFT library 0.4+
- BitsAndBytes for quantization (optional but recommended)
- Additional dependencies listed in requirements.txt

## Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/biomedical-qa-system.git
   cd biomedical-qa-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_key
   NCBI_API_KEY=your_ncbi_key
   GOOGLE_API_KEY=your_google_key
   NVIDIA_API_KEY=your_nvidia_key
   ```

5. Verify the installation:
   ```bash
   python -c "from qa_handler import BiomedicalQAHandler; print('Setup successful')"
   ```

## Configuration Options

The system uses a `config.json` file for configuring various aspects of its behavior. Here's a detailed breakdown of the configuration sections:

### Model Configuration

```json
"model": {
    "base_model": "dmis-lab/biobert-large-cased-v1.1-squad",
    "max_length": 512,
    "temperature": 0.1,
    "top_p": 0.9,
    "device": "auto"
}
```

- `base_model`: Hugging Face model ID or local path to the model
- `max_length`: Maximum sequence length for tokenizer
- `temperature`: Sampling temperature (lower = more deterministic)
- `top_p`: Nucleus sampling parameter (controls randomness)
- `device`: Computing device ("auto", "cpu", "cuda")

### Training Configuration

```json
"training": {
    "peft_method": "lora",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "batch_size": 2,
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "save_steps": 100,
    "gradient_accumulation_steps": 4,
    "layer_freezing": false,
    "num_unfrozen_layers": 0,
    "seed": 42
}
```

- `peft_method`: Parameter-Efficient Fine-Tuning method (currently only "lora")
- `lora_r`: Rank of the LoRA update matrices
- `lora_alpha`: Scaling factor for LoRA updates
- `lora_dropout`: Dropout probability for LoRA layers
- `batch_size`: Training batch size
- `learning_rate`: Learning rate for optimizer
- `num_epochs`: Number of training epochs
- `evaluation_strategy`: When to evaluate ("steps", "epoch", "no")
- `eval_steps`: How often to evaluate when strategy is "steps"
- `save_steps`: How often to save checkpoints
- `gradient_accumulation_steps`: Number of steps to accumulate gradients
- `layer_freezing`: Whether to freeze some layers
- `num_unfrozen_layers`: Number of layers to leave unfrozen
- `seed`: Random seed for reproducibility

### PubMed Configuration

```json
"pubmed": {
    "max_citations": 3,
    "timeout_long": 15,
    "timeout_short": 10,
    "max_retries": 3,
    "prioritize_reviews": true,
    "max_age_years": 5
}
```

- `max_citations`: Maximum number of citations to return
- `timeout_long`: Timeout in seconds for longer operations
- `timeout_short`: Timeout in seconds for shorter operations
- `max_retries`: Number of retries for failed requests
- `prioritize_reviews`: Whether to prioritize review articles
- `max_age_years`: Maximum age of articles to retrieve (0 for no limit)

### Citation Extraction Configuration

```json
"citation_extraction": {
    "entity_extraction_model": "gpt-4o",
    "query_generation_model": "gpt-4o",
    "extraction_temperature": 0.1,
    "generation_temperature": 0.2,
    "extraction_max_tokens": 500,
    "generation_max_tokens": 500,
    "use_llm_extraction": true,
    "use_llm_query_generation": true
}
```

- `entity_extraction_model`: Model to use for entity extraction
- `query_generation_model`: Model to use for query generation
- `extraction_temperature`: Temperature for entity extraction
- `generation_temperature`: Temperature for query generation
- `extraction_max_tokens`: Maximum tokens for entity extraction
- `generation_max_tokens`: Maximum tokens for query generation
- `use_llm_extraction`: Whether to use LLM for entity extraction
- `use_llm_query_generation`: Whether to use LLM for query generation

### Network Configuration

```json
"network": {
    "use_proxy": true,
    "http_proxy": "http://username:password@proxy.example.com:8080",
    "https_proxy": "http://username:password@proxy.example.com:8080",
    "timeout_retry_multiplier": 1.5,
    "log_network_requests": true
}
```

- `use_proxy`: Whether to use proxies for network requests
- `http_proxy`: HTTP proxy URL
- `https_proxy`: HTTPS proxy URL
- `timeout_retry_multiplier`: Factor to increase timeout on retries
- `log_network_requests`: Whether to log network requests

### Inference Configuration

```json
"inference": {
    "highlight_keywords": false,
    "include_citations": true,
    "max_new_tokens": 350
}
```

- `highlight_keywords`: Whether to highlight keywords in responses
- `include_citations`: Whether to include citations in responses
- `max_new_tokens`: Maximum number of tokens to generate

### Fallback Configuration

```json
"fallback": {
    "models": [
        "gpt-4o",
        "nvidia/meta-llama-3.1-405b-instruct",
        "gemini-1.5-pro-latest"
    ],
    "nvidia_base_url": "https://integrate.api.nvidia.com/v1"
}
```

- `models`: List of fallback models to try in order
- `nvidia_base_url`: Base URL for NVIDIA API

## Training Process

### Step 1: Prepare Training Data

Training data should be prepared in the following format:

```json
[
  {
    "question": "What genes are associated with Parkinson's disease?",
    "answer": "• SNCA\n• LRRK2\n• PARK7\n• PINK1\n• PRKN\n• GBA\n• VPS35",
    "query_type": "gene"
  },
  {
    "question": "What proteins are involved in the JAK-STAT pathway?",
    "answer": "• JAK1\n• JAK2\n• JAK3\n• TYK2\n• STAT1\n• STAT2\n• STAT3\n• STAT4\n• STAT5A\n• STAT5B\n• STAT6",
    "query_type": "protein"
  }
]
```

Save your training data as `training_data.json` and validation data as `validation_data.json` in the `data` directory.

### Step 2: Prepare the Training Script

Create a training script (`train.py`) that loads the model and applies LoRA:

```python
import json
import os
from transformers import Trainer, TrainingArguments
from model import BiomedicalModel
from utils import load_config, load_environment_variables

# Load environment variables and configuration
load_environment_variables()
config = load_config("config.json")

# Initialize the model
model = BiomedicalModel(config)
model.load_model()
model.apply_peft()  # Apply LoRA for finetuning

# Load dataset
with open("data/training_data.json", "r") as f:
    train_data = json.load(f)
with open("data/validation_data.json", "r") as f:
    eval_data = json.load(f)

# Process data
# [Code to convert data into appropriate format for the Trainer]

# Initialize training arguments
training_args = TrainingArguments(
    output_dir="output/peft_adapter",
    per_device_train_batch_size=config["training"]["batch_size"],
    per_device_eval_batch_size=config["training"]["batch_size"],
    learning_rate=config["training"]["learning_rate"],
    num_train_epochs=config["training"]["num_epochs"],
    evaluation_strategy=config["training"]["evaluation_strategy"],
    eval_steps=config["training"]["eval_steps"],
    save_steps=config["training"]["save_steps"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    seed=config["training"]["seed"],
    # Add other relevant parameters
)

# Initialize trainer
trainer = Trainer(
    model=model.peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # Add other relevant parameters
)

# Train the model
trainer.train()

# Save the adapter
model.save_peft_adapter("output/peft_adapter")
```

### Step 3: Run Training

Execute the training script:

```bash
python train.py
```

Monitor the training progress in the console. The script will create checkpoints in the `output/peft_adapter` directory.

### Step 4: Evaluate the Model

After training, evaluate the model on a separate test set:

```python
# Load test dataset
with open("data/test_data.json", "r") as f:
    test_data = json.load(f)

# Process test data
# [Code to convert data into appropriate format for evaluation]

# Load the model with the adapter
model = BiomedicalModel(config)
model.load_model()
model.load_peft_adapter("output/peft_adapter")

# Evaluate
results = []
for example in test_data:
    question = example["question"]
    answer = model.generate_answer(question)
    results.append({
        "question": question,
        "generated_answer": answer,
        "reference_answer": example["answer"]
    })

# Calculate metrics
from utils import evaluate_scientific_answers
metrics = evaluate_scientific_answers(
    [r["generated_answer"] for r in results],
    [r["reference_answer"] for r in results]
)
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

## Running the System

### Interactive Mode

The system can be run in interactive mode using the following command:

```bash
python run.py --interactive
```

This will start an interactive session where you can ask questions and get responses.

### API Mode

The system can also be run as an API service:

```bash
python run.py --api --port 8000
```

This will start a web server on port 8000 that accepts POST requests to the `/query` endpoint.

Example API request:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What genes are associated with Parkinson's disease?", "include_citations": true}'
```

### Batch Mode

For processing multiple questions in batch mode:

```bash
python run.py --batch questions.txt --output results.json
```

Where `questions.txt` contains one question per line, and `results.json` will contain the answers.

## Evaluating Performance

The system provides several ways to evaluate performance:

### Accuracy Metrics

Use the `evaluate_scientific_answers` function from `utils.py` to calculate precision, recall, and F1 score for structured answers:

```python
from utils import evaluate_scientific_answers

metrics = evaluate_scientific_answers(generated_answers, reference_answers)
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

### Citation Quality Assessment

Evaluate the quality of citations by examining:

1. **Citation Relevance**: How relevant are the returned citations to the query?
2. **Citation Coverage**: Do the citations cover all aspects of the query?
3. **Citation Recency**: Are the citations recent enough to provide up-to-date information?

You can create a script to evaluate citation quality:

```python
from qa_handler import BiomedicalQAHandler
from utils import load_config

# Load configuration
config = load_config("config.json")

# Initialize QA handler
qa_handler = BiomedicalQAHandler(config)
qa_handler.setup()

# Test questions
questions = [
    "What genes are associated with Parkinson's disease?",
    "What proteins are involved in the JAK-STAT pathway?",
    # Add more test questions
]

# Evaluate citation quality
results = []
for question in questions:
    result = qa_handler.answer_question(question, include_citations=True)
    citations = result["citations"]
    
    # Calculate metrics
    num_citations = len(citations)
    num_reviews = sum(1 for c in citations if c.get("is_review", False))
    avg_year = sum(int(c.get("year", 2020)) for c in citations) / max(1, num_citations)
    
    results.append({
        "question": question,
        "num_citations": num_citations,
        "percent_reviews": num_reviews / max(1, num_citations) * 100,
        "avg_year": avg_year,
        "citations": citations
    })

# Print summary
for r in results:
    print(f"Question: {r['question']}")
    print(f"  Citations: {r['num_citations']}")
    print(f"  Reviews: {r['percent_reviews']:.1f}%")
    print(f"  Avg Year: {r['avg_year']:.1f}")
    print()
```

### Performance Benchmarking

Measure system performance metrics:

```python
import time
from qa_handler import BiomedicalQAHandler
from utils import load_config

# Load configuration
config = load_config("config.json")

# Initialize QA handler
qa_handler = BiomedicalQAHandler(config)
qa_handler.setup()

# Test questions
questions = [
    "What genes are associated with Parkinson's disease?",
    "What proteins are involved in the JAK-STAT pathway?",
    # Add more test questions
]

# Benchmark performance
results = []
for question in questions:
    start_time = time.time()
    result = qa_handler.answer_question(question, include_citations=True)
    end_time = time.time()
    
    results.append({
        "question": question,
        "processing_time": end_time - start_time,
        "answer_source": result["answer_source"]
    })

# Print summary
avg_time = sum(r["processing_time"] for r in results) / len(results)
primary_count = sum(1 for r in results if r["answer_source"] == "primary_model")
fallback_count = len(results) - primary_count

print(f"Average processing time: {avg_time:.2f} seconds")
print(f"Primary model success rate: {primary_count / len(results) * 100:.1f}%")
print(f"Fallback usage rate: {fallback_count / len(results) * 100:.1f}%")
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Failures

**Symptoms**: Error messages like "Failed to load model" or CUDA out-of-memory errors.

**Solutions**:
- Check if you have enough GPU memory
- Try using a smaller model
- Enable 4-bit quantization
- Ensure you have the correct model name in config.json

#### 2. Citation Fetching Issues

**Symptoms**: "No citations found" or timeout errors when fetching citations.

**Solutions**:
- Check your network connection
- Verify your NCBI API key in .env
- Adjust timeout settings in config.json
- Configure proxy settings if you're behind a firewall

#### 3. Finetuning Problems

**Symptoms**: Training fails or model performance doesn't improve.

**Solutions**:
- Check your training data format
- Adjust LoRA parameters (r, alpha, etc.)
- Try a different learning rate
- Increase batch size or use gradient accumulation
- Check for NaN values during training

#### 4. Fallback Model Issues

**Symptoms**: Fallback models not working or returning errors.

**Solutions**:
- Check API keys in .env
- Verify fallback model names in config.json
- Ensure network connectivity to API services
- Check API usage limits

### Logs and Debugging

The system creates detailed logs in:
- `biomedical_qa.log` - Main system log
- `citations.log` - Citation fetching log

To enable debug-level logging, modify the logging setup in each module:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("biomedical_qa.log"),
        logging.StreamHandler()
    ]
)
```

For advanced debugging, you can use Python's built-in debugger or tools like VS Code's debugger.

## Additional Resources

- **Hugging Face Documentation**: For transformer models and tokenizers
- **PEFT Documentation**: For LoRA and other parameter-efficient fine-tuning methods
- **PubMed API Documentation**: For understanding citation fetching
- **OpenAI/NVIDIA/Google API Documentation**: For fallback models

---

For more information, please contact the system maintainers or refer to the project repository.

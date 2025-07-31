# Biomedical QA System - Fallback Inference Guide

This guide explains how to use the direct fallback inference functionality that bypasses the trained model and uses OpenAI's API directly for answer generation.

## Overview

The system now supports two modes of operation:

1. **Standard Mode**: Uses the fine-tuned biomedical model for inference
2. **Direct Fallback Mode**: Loads the model but uses OpenAI's API (GPT-4o/GPT-3.5-turbo) for answer generation

## Configuration

The fallback behavior is controlled by the `use_direct_fallback` parameter in `config.json`:

```json
{
  "fallback": {
    "models": [
        "gpt-4o",
        "gpt-3.5-turbo"
    ],
    "use_direct_fallback": true
  },
  "inference": {
    "use_llm_parent": false
  }
}
```

### Key Parameters:

- `fallback.use_direct_fallback`: Set to `true` to enable direct fallback mode
- `inference.use_llm_parent`: Should be `false` to bypass the primary model
- `fallback.models`: List of OpenAI models to use as fallback

**To switch response sources**: Change `use_llm_parent` to `true` in the config to get responses from your fine-tuned model, or set it to `false` to get responses from the OpenAI fallback models.

## Usage

### 1. Training (Unchanged)

Training proceeds normally regardless of the fallback setting:

```bash
python train.py
```

The model will be trained and saved to `output/peft_adapter/` as usual.

### 2. Testing with Fallback

The existing test script now automatically detects the fallback setting:

```bash
python test_trained_model.py
```

- If `use_direct_fallback` is `true`: Uses fallback pipeline
- If `use_direct_fallback` is `false`: Uses trained model

### 3. Dedicated Inference Script

Use the new inference script for more control:

```bash
# Run test questions with fallback
python inference_with_fallback.py --mode test

# Interactive session with fallback
python inference_with_fallback.py --mode interactive
```

### 4. Direct Fallback Test

Run the original direct fallback test:

```bash
python tests/test_direct_fallback_live.py
```

## How It Works

### Standard Mode (`use_direct_fallback: false`)
1. Loads the base model (BioGPT)
2. Loads the fine-tuned LoRA adapter
3. Generates answers using the trained model
4. Falls back to OpenAI only if the model fails

### Direct Fallback Mode (`use_direct_fallback: true`)
1. Loads the model (for completeness)
2. Bypasses model inference entirely
3. Uses OpenAI API directly for all answer generation
4. Still fetches citations from PubMed
5. Provides the same structured output format

## Benefits of Direct Fallback Mode

1. **Consistency**: Uses state-of-the-art models (GPT-4o) for reliable answers
2. **Speed**: No local model inference overhead
3. **Quality**: Leverages OpenAI's latest biomedical knowledge
4. **Flexibility**: Can switch between modes without retraining
5. **Development**: Useful for testing and comparison

## Environment Setup

Ensure you have the OpenAI API key set:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or add it to your `.env` file:

```
OPENAI_API_KEY=your-api-key-here
```

## Output Format

Both modes provide the same output structure:

```python
{
    "question": "What genes are associated with Parkinson's disease?",
    "answer": "• SNCA\n• LRRK2\n• PARK7\n...",
    "formatted_answer": "Answer with citations...",
    "query_type": "gene",
    "citations": [...],
    "answer_source": "direct_fallback_gpt-4o",  # or "primary_model"
    "processing_time_seconds": 2.34,
    "model_status": "loaded_but_using_fallback",  # Additional info
    "adapter_loaded": true
}
```

## Switching Between Modes

To switch modes, simply change the config and restart:

1. Edit `config.json`
2. Set `fallback.use_direct_fallback` to `true` or `false`
3. Run your inference script

No retraining required!

## Troubleshooting

### Common Issues:

1. **OpenAI API Key Missing**: Ensure `OPENAI_API_KEY` is set
2. **Network Issues**: Check internet connection for API calls
3. **Rate Limits**: OpenAI API has rate limits; the system includes retry logic
4. **Model Loading**: Even in fallback mode, the system tries to load the model for completeness

### Logs:

Check the logs for detailed information:
- `logs/inference_fallback.log` - Inference script logs
- `biomedical_qa.log` - General system logs

## Example Usage

```python
from inference_with_fallback import FallbackInferenceHandler
from src.utils import load_config, load_environment_variables

# Setup
load_environment_variables()
config = load_config("config.json")

# Initialize handler
handler = FallbackInferenceHandler(config)
handler.setup_with_fallback_inference()

# Ask question
result = handler.generate_answer_with_fallback(
    "What genes are associated with Parkinson's disease?",
    include_citations=True
)

print(result["formatted_answer"])
```

This approach gives you the flexibility to use either the trained model or the fallback pipeline while maintaining the same interface and output format.

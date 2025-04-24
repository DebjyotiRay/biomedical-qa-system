"""
Model management module for Biomedical QA system (Generative Focus)
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, # Using Causal LM
    PreTrainedTokenizer,
    PreTrainedModel,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)

# Import constants and utils needed
from src.constants import INFERENCE_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, QUERY_TYPE_INSTRUCTIONS
from src.utils import determine_query_type, format_scientific_answer

# Set up logging
logger = logging.getLogger(__name__)

class BiomedicalModel:
    """Handles the biomedical language model setup and inference (Generative)"""

    def __init__(self, config: Dict[str, Any]):
        """ Initialize the model manager """
        self.config = config
        self.model_config = config.get("model", {})
        self.inference_config = config.get("inference", {})
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.peft_model: Optional[PeftModel] = None
        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        """ Determine the appropriate device (CPU/GPU/MPS) """
        device_pref = self.model_config.get("device", "auto").lower()

        if device_pref == "cpu":
            return torch.device("cpu")
        elif device_pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        # elif device_pref == "mps" and torch.backends.mps.is_available():
        #     return torch.device("mps") # MPS support can be experimental

        # Auto detection
        if torch.cuda.is_available():
            return torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #      logger.warning("MPS available but support might be limited. Using MPS.")
        #      return torch.device("mps")
        else:
            logger.warning("CUDA (and MPS) not available. Using CPU for model inference.")
            return torch.device("cpu")

    def load_model(self) -> bool:
        """ Load the pre-trained generative model and tokenizer """
        model_name = self.model_config.get("base_model", "microsoft/BioGPT-Large")
        logger.info(f"Loading model: {model_name}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                # Some models like GPT2/BioGPT don't have a PAD token, using EOS is common
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Tokenizer pad_token set to eos_token ({self.tokenizer.eos_token})")


            # Configure proxy properly
            # Set proxy correctly based on environment variables, not code
            # The error showed that proxy was set incorrectly with code like os.environ.get(...)
            
            # Configure quantization only if on CUDA
            quant_config = None 
            model_kwargs = {}
            if torch.cuda.is_available() and self.device.type == "cuda":
                logger.info("Using 4-bit quantization for GPU efficiency")
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                model_kwargs["quantization_config"] = quant_config
                model_kwargs["device_map"] = "auto" # Let HF handle distribution
                # Ensure torch_dtype is set for quantized models
                model_kwargs["torch_dtype"] = torch.float16

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            # Move to device if not using device_map (i.e., on CPU or MPS)
            if "device_map" not in model_kwargs and self.model:
                 self.model.to(self.device)

            # Prepare for k-bit training if quantization is used
            # This needs to happen *after* loading the base model but *before* PEFT application if training
            if quant_config and self.model:
                # We prepare it here, but enable gradient checkpointing just before training
                self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)


            logger.info(f"Model '{model_name}' loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {e}", exc_info=True) # Add traceback
            return False

    def load_peft_adapter(self, adapter_path: Optional[str] = None) -> bool:
        """ Load a trained PEFT adapter if available """
        if self.model is None:
            logger.error("Base model must be loaded before loading PEFT adapter")
            return False

        if adapter_path is None:
            adapter_path = os.path.join(
                self.config.get("output", {}).get("dir", "output"),
                "peft_adapter" # Default adapter save location
            )

        try:
            if os.path.exists(adapter_path):
                logger.info(f"Loading PEFT adapter from {adapter_path}")
                self.peft_model = PeftModel.from_pretrained(
                    self.model,
                    adapter_path,
                    is_trainable=False # Load for inference
                )
                # No need to move PEFT model explicitly if base model uses device_map="auto"
                # If not using device_map, ensure it's on the right device:
                # if self.model_config.get("device", "auto") != "auto":
                #      self.peft_model.to(self.device)
                logger.info("PEFT adapter loaded successfully for inference")
                return True
            else:
                logger.warning(f"PEFT adapter not found at {adapter_path}. Using base model.")
                return False
        except Exception as e:
            logger.error(f"Error loading PEFT adapter from {adapter_path}: {e}", exc_info=True)
            return False

    def apply_peft(self) -> bool:
        """ Apply Parameter-Efficient Fine-Tuning (LoRA) for Training """
        if self.model is None:
            logger.error("Model must be loaded before applying PEFT")
            return False

        try:
            logger.info("Applying LoRA for parameter-efficient fine-tuning")

            # Define target modules based on model type
            model_type = self.model_config.get("base_model", "").lower()
            target_modules = []
            # --- Determine Target Modules ---
            # This part requires knowing the architecture. Inspect the model or check documentation.
            # Examples:
            if "biogpt" in model_type:
                 # BioGPT's structure might differ, common targets could be attention layers
                 # Inspect self.model.named_modules() to find likely candidates
                 # Example: target_modules = ["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"] # Common in GPT-like
                 target_modules = ["q_proj", "v_proj"] # Simplified common targets
                 logger.info(f"Using target modules for BioGPT-like: {target_modules}")
            elif "llama" in model_type:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                logger.info(f"Using target modules for Llama-like: {target_modules}")
            elif "gpt-neo" in model_type or "gpt-j" in model_type:
                 target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out"]
                 logger.info(f"Using target modules for GPT-Neo/J-like: {target_modules}")
            else:
                logger.warning(f"LoRA target modules not explicitly defined for {model_type}. Attempting common defaults.")
                # Fallback - check common names
                module_names = {name for name, _ in self.model.named_modules()}
                if "q_proj" in module_names and "v_proj" in module_names: target_modules = ["q_proj", "v_proj"]
                elif "query" in module_names and "value" in module_names: target_modules = ["query", "value"]
                if not target_modules: logger.error("Could not determine default LoRA target modules. Training might fail.")


            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.get("training", {}).get("lora_r", 8),
                lora_alpha=self.config.get("training", {}).get("lora_alpha", 16),
                lora_dropout=self.config.get("training", {}).get("lora_dropout", 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules if target_modules else None # Pass None if empty to let PEFT try defaults
            )

            # Apply LoRA
            # Ensure model is prepared *before* getting peft model if quantized
            # Enable gradient checkpointing here before wrapping with PEFT
            if self.model.supports_gradient_checkpointing and self.device.type == "cuda":
                 logger.info("Enabling gradient checkpointing for training.")
                 # prepare_model_for_kbit_training should be called *before* this if using quantization
                 self.model.gradient_checkpointing_enable()
                 # Re-call prepare_model_for_kbit_training specifically for gradient checkpointing compatibility
                 self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)


            self.peft_model = get_peft_model(self.model, lora_config)
            self.peft_model.print_trainable_parameters()
            return True
        except Exception as e:
            logger.error(f"Error applying PEFT: {e}", exc_info=True)
            return False

    def design_prompt(self, question: str, query_type: Optional[str] = None) -> str:
        """ Design an optimized prompt for biomedical question answering """
        if query_type is None:
            query_type = determine_query_type(question)
        type_specific = QUERY_TYPE_INSTRUCTIONS.get(query_type, QUERY_TYPE_INSTRUCTIONS["general"])
        prompt = f"{INFERENCE_SYSTEM_PROMPT.strip()}\n\n"
        prompt += f"{type_specific.strip()}\n\n"
        prompt += f"Question: {question}\n\n"
        if query_type in ['gene', 'protein', 'pathway', 'disease'] and FEW_SHOT_EXAMPLES:
             prompt += f"{FEW_SHOT_EXAMPLES.strip()}\n\n"
        prompt += "Answer:"
        return prompt

    def generate_answer(self, question: str, query_type: Optional[str] = None) -> str:
        """ Generate a scientific answer using the loaded generative model """
        if self.tokenizer is None or (self.model is None and self.peft_model is None):
            logger.error("Model or tokenizer not loaded.")
            return "Error: Model/Tokenizer not loaded."

        try:
            prompt = self.design_prompt(question, query_type)
            if not self.tokenizer: raise RuntimeError("Tokenizer is not initialized.")

            # Calculate max prompt length to leave room for generation
            max_length = self.model_config.get("max_length", 512)
            max_new_tokens = self.inference_config.get("max_new_tokens", 350)
            max_prompt_len = max_length - max_new_tokens - 5 # Subtract buffer for special tokens

            if max_prompt_len <= 0:
                 logger.error(f"max_length ({max_length}) too small for max_new_tokens ({max_new_tokens}).")
                 return "Error: Configuration max_length too small."


            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_length = inputs["input_ids"].shape[1]

            active_model = self.peft_model if self.peft_model is not None else self.model
            if not active_model: raise RuntimeError("No active model found.")

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": self.model_config.get("temperature", 0.1),
                "top_p": self.model_config.get("top_p", 0.9),
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            # Prevent temp=0 with do_sample=True
            if gen_kwargs["temperature"] <= 0: gen_kwargs["temperature"] = 0.01

            logger.debug(f"Generating answer with kwargs: {gen_kwargs}")

            with torch.no_grad():
                if input_length == 0:
                    logger.error("Input prompt truncated to zero length.")
                    return "Error: Input prompt too long after truncation."

                outputs = active_model.generate(**inputs, **gen_kwargs)

            if outputs is None or outputs.shape[1] <= input_length:
                 logger.warning("Model generation returned empty or unchanged output.")
                 return "Error: Model generation failed."

            generated_ids = outputs[0][input_length:]
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            formatted_answer = format_scientific_answer(answer.strip())

            if not formatted_answer or formatted_answer.startswith("Answer:") or len(formatted_answer) < 3:
                 logger.warning(f"Model generated empty or invalid answer: '{answer}' -> '{formatted_answer}'")
                 raw_answer_debug = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                 logger.debug(f"Raw generated tokens (decoded): {raw_answer_debug}")
                 return "Error: Model failed to generate a valid structured answer."

            return formatted_answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            return "Error: An exception occurred during answer generation."

    def highlight_keywords(self, text: str) -> str:
        """ Highlight keywords - less critical for structured lists """
        if "•" in text or text.strip().startswith("-"): return text
        return text # Keep simple

    def save_peft_adapter(self, path: Optional[str] = None) -> bool:
        """ Save the PEFT adapter model """
        if self.peft_model is None:
            logger.error("No PEFT model to save")
            return False
        if path is None:
            path = os.path.join(self.config.get("output", {}).get("dir", "output"), "peft_adapter")
        try:
            os.makedirs(path, exist_ok=True)
            self.peft_model.save_pretrained(path)
            if self.tokenizer: self.tokenizer.save_pretrained(path)
            logger.info(f"PEFT adapter and tokenizer saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving PEFT adapter: {e}", exc_info=True)
            return False

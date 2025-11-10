import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM,
    AutoModelForSeq2SeqLM, T5Tokenizer
)

from peft import PeftModel
from app.config import BASE_MODEL_PATH, LORA_ADAPTER_PATH, MAX_LEN
import os
import logging

class ModelLoader:
    def __init__(self):
        logger = logging.getLogger(__name__)
        try:
            
            if not os.path.exists(BASE_MODEL_PATH):
                logger.error(f"Base model not found at {BASE_MODEL_PATH}")
                raise FileNotFoundError(f"Base model not found at {BASE_MODEL_PATH}")
            if not os.path.exists(LORA_ADAPTER_PATH):
                logger.error(f"LoRA adapter not found at {LORA_ADAPTER_PATH}")
                raise FileNotFoundError(f"LoRA adapter not found at {LORA_ADAPTER_PATH}")
            
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
            
           
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            
            try:
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    BASE_MODEL_PATH,
                    num_labels=1  # Binary classification with single output
                )
            except Exception as e:
                logger.warning(f"Failed to load base model with num_labels=1: {e}")
                #
                base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_PATH)
            
            # Load LoRA adapter
            try:
                self.model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
            except Exception as e:
                logger.error(f"Failed to load LoRA adapter: {e}")
                raise
            self.model.to(self.device)
            self.model.eval()
            logger.info("Base model and LoRA adapter loaded successfully.")

            self.model = base_model
            
            
            try:
                self.gen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")   
                self.gen_model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen2.5-1.5B-Instruct"
                ).to(self.device)
                 
                self.gen_model.eval()
                 #logger.info("Flan-T5-small loaded successfully.")
            except Exception as e: 
                logger.error(f"Failed to load Flan-T5-small: {e}")
                raise
        except Exception as e:
            logger.error(f"ModelLoader initialization failed: {e}")
            raise

    def encode(self, title, content=None):
        """Encode title and content into model input format."""
        
        if content and str(content).strip():
            text = f"title: {title} [SEP] content: {content}"
        else:
           
            text = str(title).strip()
        
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length"
        )
        
        return {k: v.to(self.device) for k, v in encoded.items()}

    def explain_with_model(self, title: str, content: str, score: float, *,
                           max_new_tokens: int = 80,
                           temperature: float = 0.2,
                           top_p: float = 0.95,
                           detailed: bool = True) -> str:
        """
        Use Flan-T5-small to generate a natural language explanation for the
        clickbait score, taking both title and content into account.
        """
        
        messages = [
        {"role": "system", "content": "You explain briefly why a headline may be clickbait if score>0.7, else it is not clickbait, in 1–2 clear sentences."},
        {"role": "user", "content":
            f"Title: {title}\n"
            f"Content: {content}\n"
            f"Score: {score:.2f}\n\n"
            "Explain why the title is clickbait without repeating content."
        }
    ]

        prompt = self.gen_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        try:
            # T5 expects a string input (no special tokens/format needed)
            inputs = self.gen_tokenizer(
                prompt, 
                return_tensors="pt",
                max_length=MAX_LEN,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.gen_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=(temperature > 0),
                    num_beams=1 if temperature > 0 else 4,
                    
                    pad_token_id=self.gen_tokenizer.pad_token_id,
                )
            
            explanation = self.gen_tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            split_token = "assistant"
            prompt = explanation.lower()
            # Find the index
            idx = prompt.find(split_token)
            if idx != -1:
                # Advance index past "assistant" and any newlines/spaces
                answer_start = idx + len(split_token)
                answer = prompt[answer_start:].lstrip()
            else:
                answer = ""

            print(answer)
            return answer

        except Exception as e:
            print(f"Error generating explanation: {e}")
            
            if score > 0.7:
                return "This article shows strong clickbait characteristics in its title and content."
            elif score > 0.5:
                return "This article has some clickbait-like patterns but is not extreme."
            else:
                return "This article generally avoids clickbait tactics in both title and content."

import os
import torch
import transformers
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from .base_llm import BaseLLM

class PRMModel(BaseLLM):
    """
    Implementation for PRM models using Hugging Face.
    """
    
    def __init__(self, 
                 model_name: str = "peiyi9979/math-shepherd-mistral-7b-prm",
                 cache_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize the PRM LLM.
        
        Args:
            model_name (str): Name of the Qwen model to use
            cache_dir (str, optional): Directory to cache models
            **kwargs: Additional model parameters
        """
        super().__init__(model_name, **kwargs)
        
        # Set Hugging Face cache directory
        if cache_dir:
            os.environ['HF_HOME'] = cache_dir
        
        # Initialize tokenizer and model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set default parameters
        self.default_params = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_new_tokens": kwargs.get("max_new_tokens", 500),
            "top_p": kwargs.get("top_p", 0.8),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1)
        }
        
        # Initialize model with appropriate device and dtype
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = 'ки'

        self.candidate_tokens = self.tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:] # [648, 387]
        self.step_tag_id = self.tokenizer.encode(f"{self.step_tag}")[-1] # 12902

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
    def generate(self, 
                prompts: List[str], 
                temperature: float = 0.0,
                **kwargs) -> List[str]:
        """
        Generate text for a list of prompts using Flan T5 model.
        
        Args:
            prompts (List[str]): List of input prompts
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List[str]: List of generated outputs
        """
        return self.batch_generate(prompts, batch_size=len(prompts), 
                                 temperature=temperature, 
                                 **kwargs)
    
    def batch_generate(self, 
                      prompts: List[str], 
                      batch_size: int = 8,
                      **kwargs) -> List[str]:
        """
        Generate text for a list of prompts in batches using Flan T5 model.
        
        Args:
            prompts (List[str]): List of input prompts
            batch_size (int): Size of each batch
            **kwargs: Additional generation parameters
            
        Returns:
            List[str]: List of generated outputs
        """
        outputs = []
        
        # Merge default parameters with provided kwargs
        params = {**self.default_params, **kwargs}

        device = next(self.model.parameters()).device
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating text", unit="batch"):
            batch = prompts[i:i+batch_size]

            encoded_inputs = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
            encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
            # Process batch in parallel
            step_scores_list = []  # To hold the list of step scores for each output

            with torch.no_grad():
                logits = self.model(**encoded_inputs).logits[:,:,self.candidate_tokens]
                scores = logits.softmax(dim=-1)[:,:,0]
                
                for i, input_ids in enumerate(encoded_inputs['input_ids']):
                    # Get the step scores for this example
                    step_scores = scores[i][input_ids == self.step_tag_id].tolist()
                    step_scores_list.append(step_scores)
            
            outputs.extend(step_scores_list)
            
        return outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Qwen model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "model_name": self.model_name,
            "model_type": "flan-t5",
            "provider": "huggingface",
            "tokenizer": self.tokenizer.__class__.__name__,
            "device": self.model.device,
            "dtype": str(self.model.dtype),
            "default_params": self.default_params
        }
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get a list of available Qwen models.
        
        Returns:
            List[str]: List of model names
        """
        return [
            "google/flan-t5-xxl",
            "google/flan-t5-xl",
            "google/flan-t5-large",
            "google/flan-t5-base",
            "google/flan-t5-small",
        ] 
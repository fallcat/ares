import os
import torch
import transformers
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from .base_llm import BaseLLM

class QwenLLM(BaseLLM):
    """
    Implementation for Qwen models using Hugging Face.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 cache_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Qwen LLM.
        
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
        
        # Create text generation pipeline
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            **self.default_params
        )
        
    def generate(self, 
                prompts: List[str], 
                max_new_tokens: int = 500,
                temperature: float = 0.0,
                **kwargs) -> List[str]:
        """
        Generate text for a list of prompts using Qwen model.
        
        Args:
            prompts (List[str]): List of input prompts
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List[str]: List of generated outputs
        """
        return self.batch_generate(prompts, batch_size=len(prompts), 
                                 max_new_tokens=max_new_tokens, 
                                 temperature=temperature, 
                                 **kwargs)
    
    def batch_generate(self, 
                      prompts: List[str], 
                      batch_size: int = 8,
                      **kwargs) -> List[str]:
        """
        Generate text for a list of prompts in batches using Qwen model.
        
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
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating text", unit="batch"):
            batch = prompts[i:i+batch_size]
            
            try:
                # Generate for the batch
                batch_outputs = self.pipeline(
                    batch,
                    max_new_tokens=params.get("max_new_tokens", 500),
                    temperature=params.get("temperature", 0.0),
                    top_p=params.get("top_p", 0.8),
                    repetition_penalty=params.get("repetition_penalty", 1.1),
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
                
                # Extract generated text
                for output in batch_outputs:
                    generated_text = output[0]['generated_text']
                    outputs.append(generated_text)
                    
            except Exception as e:
                print(f"Error generating text: {e}")
                # Return empty strings for failed generations
                outputs.extend(["" for _ in range(len(batch))])
                
        return outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Qwen model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "model_name": self.model_name,
            "model_type": "qwen",
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
            "Qwen/Qwen3-4B",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-110B-Instruct",
            "Qwen/Qwen1.5-7B-Chat",
            "Qwen/Qwen1.5-14B-Chat",
            "Qwen/Qwen1.5-32B-Chat",
            "Qwen/Qwen1.5-72B-Chat",
            "Qwen/Qwen1.5-110B-Chat"
        ] 
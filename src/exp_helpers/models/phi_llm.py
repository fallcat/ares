import os
import torch
import transformers
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from .base_llm import BaseLLM
from diskcache import Cache

cache = Cache(os.getenv("CACHE_DIR"))

class PhiLLM(BaseLLM):
    """
    Implementation of the Phi-4 model from Microsoft using Hugging Face.
    """
    
    def __init__(self, model_name: str = "microsoft/phi-4", **kwargs):
        """
        Initialize the Phi LLM.
        
        Args:
            model_name (str): Name of the model (default: microsoft/phi-4)
            **kwargs: Additional model parameters
        """
        super().__init__(model_name, **kwargs)
        
        # Set Hugging Face cache directory
        os.environ['HF_HOME'] = kwargs.get('cache_dir', '/shared_data0/hf_cache')
        
        # Initialize tokenizer and model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False
        )
        
    def generate(self, 
                prompts: List[str], 
                max_new_tokens: int = 500,
                temperature: float = 0.0,
                **kwargs) -> List[str]:
        """
        Generate text for a list of prompts using Phi model.
        
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
        Generate text for a list of prompts in batches using Phi model.
        
        Args:
            prompts (List[str]): List of input prompts
            batch_size (int): Size of each batch
            **kwargs: Additional generation parameters
            
        Returns:
            List[str]: List of generated outputs
        """
        outputs = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating text", unit="batch"):
            batch = prompts[i:i+batch_size]
            batch_outputs = self.model(batch, **kwargs)
            
            for bi, output in enumerate(batch_outputs):
                try:
                    generated_text = output[bi]['generated_text'].replace(batch[bi], '')
                    outputs.append(generated_text)
                except Exception as e:
                    import pdb; pdb.set_trace()
                    print(f"Error generating text for prompt {batch[bi]}: {e}")
                    outputs.append(batch[bi])
                
        return outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Phi model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "model_name": self.model_name,
            "model_type": "phi",
            "tokenizer": self.tokenizer.__class__.__name__,
            "device": self.model.device,
            "dtype": str(self.model.model.dtype)
        }

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get a list of available Phi models.
        
        Returns:
            List[str]: List of model names
        """
        return [
            "microsoft/phi-4",
            "microsoft/phi-3.5-turbo",
            "microsoft/phi-3.5-turbo-instruct"
        ]
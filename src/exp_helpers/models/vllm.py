import os
import torch
import transformers
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from .base_llm import BaseLLM
from transformers import AutoTokenizer

class VLLM(BaseLLM):
    """
    Implementation for Qwen models using Hugging Face.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-4B",
                 cache_dir: Optional[str] = None,
                 enable_thinking: bool = False,
                 max_model_len: int = 10000,
                 use_tqdm: bool = False,
                 **kwargs):
        """
        Initialize the Qwen LLM.
        
        Args:
            model_name (str): Name of the Qwen model to use
            cache_dir (str, optional): Directory to cache models
            **kwargs: Additional model parameters
        """
        super().__init__(model_name, **kwargs)
        print("model_name", model_name)
        
        # Set Hugging Face cache directory
        if cache_dir:
            os.environ['HF_HOME'] = cache_dir
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            # trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token # Ensure pad token is set
        
        # Set default parameters
        self.default_params = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_new_tokens": kwargs.get("max_new_tokens", 500),
            "top_p": kwargs.get("top_p", 0.8),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1)
        }
        
        # Initialize vLLM
        # vLLM can load PEFT adapters directly or load a merged model.
        # Loading the adapter with the base model is generally preferred.
        print(f"Initializing vLLM with base model {model_name}")

        self.model = LLM(
            model=model_name,
            tokenizer=model_name, # Use the tokenizer saved with the adapter
            tensor_parallel_size=1,
            max_model_len=max_model_len,
        )
        print("vLLM initialized.")
        self.enable_thinking = enable_thinking
        self.use_tqdm = use_tqdm
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
                      max_new_tokens: int = 500,
                      temperature: float = 0.0,
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
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.80,
            top_k=20,
            min_p=0,
            presence_penalty=1.5,
            max_tokens=max_new_tokens,
            # stop=config.VLLM_STOP_SEQUENCES,
            # Add logprobs=1 to potentially get probabilities if needed for analysis,
            # but not required for simple prediction extraction.
            # logprobs=1,
        )

        new_prompts = []
        for prompt in prompts:
            chat_format_prompt = self.tokenizer.apply_chat_template([
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ], tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
            new_prompts.append(chat_format_prompt)

        raw_outputs = self.model.generate(new_prompts, sampling_params=sampling_params, use_tqdm=self.use_tqdm)

        outputs = []
        for output in raw_outputs:
            generated_text = output.outputs[0].text.strip()
            if '</think>' in generated_text:
                generated_text = generated_text.split('</think>')[1].strip()
            outputs.append(generated_text)
            
                
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
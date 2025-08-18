import os
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm
import openai
from .base_llm import BaseLLM
from diskcache import Cache
import concurrent.futures

# cache = Cache(os.getenv("CACHE_DIR"))

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Move this function outside the class to avoid cache issues with 'self'
# @cache.memoize()
def cached_openai_generate(model_name, prompt, **kwargs) -> str:
    """
    Generate text for a single prompt using OpenAI model.
    This is a standalone function to work better with caching.
    """
    output = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        **kwargs
    ).choices[0].message.content
    return output

def non_cached_openai_generate(model_name, prompt, temperature=0.0, **kwargs) -> str:
    """
    Generate text for a single prompt using OpenAI model.
    This is a standalone function to work better with caching.
    """
    output = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        **kwargs
    ).choices[0].message.content
    return output

class OpenAILLM(BaseLLM):
    """
    Implementation for OpenAI models (GPT-3.5, GPT-4, GPT-4o-mini, etc.).
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize the OpenAI LLM.
        
        Args:
            model_name (str): Name of the OpenAI model to use
            api_key (str, optional): OpenAI API key. If not provided, will look for OPENAI_API_KEY env var
            **kwargs: Additional model parameters
        """
        super().__init__(model_name, **kwargs)
        
        # Set up API key
        if api_key is not None:
            os.environ["OPENAI_API_KEY"] = api_key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
            
        # Initialize OpenAI client
        self.client = openai_client
        
        # Set default parameters
        self.default_params = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 500),
            "top_p": kwargs.get("top_p", 1.0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0)
        }
        
    def generate(self, 
                prompts: List[str], 
                max_new_tokens: int = 500,
                temperature: float = 0.0,
                **kwargs) -> List[str]:
        """
        Generate text for a list of prompts using OpenAI model.
        
        Args:
            prompts (List[str]): List of input prompts
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List[str]: List of generated outputs
        """
        # Convert max_new_tokens to max_tokens for OpenAI API
        kwargs["max_tokens"] = max_new_tokens
        
        return self.batch_generate(prompts, batch_size=len(prompts), 
                                 temperature=temperature, 
                                 **kwargs)
    
    def single_generate(self, 
                        prompt: str, 
                        **kwargs) -> str:
        """
        Generate text for a single prompt using OpenAI model.
        Uses the cached function for actual generation.
        """
        if kwargs.get("temperature", None) is not None and kwargs.get("temperature", 0.0) != 0.0:
            return non_cached_openai_generate(self.model_name, prompt, **kwargs)
        else:
            return cached_openai_generate(self.model_name, prompt, **kwargs)

    def batch_generate(self, 
                      prompts: List[str], 
                      batch_size: int = 8,
                      verbose: bool = False,
                      temperature: Optional[float] = None, 
                      **kwargs) -> List[str]:
        """
        Generate text for a list of prompts in batches using OpenAI model.
        
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
        if temperature is not None:
            params["temperature"] = temperature
        
        # Convert max_new_tokens to max_tokens if present
        if "max_new_tokens" in params:
            params["max_tokens"] = params.pop("max_new_tokens")
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating text", unit="batch", disable=not verbose):
            batch = prompts[i:i+batch_size]

            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [executor.submit(self.single_generate, prompt, **params) for prompt in batch]
                for f in futures: # this is to avoid the issue of the futures not being completed in order
                    try:
                        response_text = f.result()
                        outputs.append(response_text)
                    except Exception as e:
                        print(f"Error generating text: {e}")
                        import pdb; pdb.set_trace()
                        outputs.append("") 
        return outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the OpenAI model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "model_name": self.model_name,
            "model_type": "openai",
            "provider": "openai",
            "default_params": self.default_params
        }
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get a list of available OpenAI models.
        
        Returns:
            List[str]: List of model names
        """
        return [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4o-max"
        ] 
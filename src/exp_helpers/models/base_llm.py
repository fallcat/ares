from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseLLM(ABC):
    """
    Abstract base class for all LLM implementations.
    Defines the interface that all LLM models must implement.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the base LLM.
        
        Args:
            model_name (str): Name of the model
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.model_kwargs = kwargs
        
    @abstractmethod
    def generate(self, 
                prompts: List[str], 
                max_new_tokens: int = 500,
                temperature: float = 0.0,
                **kwargs) -> List[str]:
        """
        Generate text for a list of prompts.
        
        Args:
            prompts (List[str]): List of input prompts
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List[str]: List of generated outputs
        """
        pass
    
    @abstractmethod
    def batch_generate(self, 
                      prompts: List[str], 
                      batch_size: int = 8,
                      **kwargs) -> List[str]:
        """
        Generate text for a list of prompts in batches.
        
        Args:
            prompts (List[str]): List of input prompts
            batch_size (int): Size of each batch
            **kwargs: Additional generation parameters
            
        Returns:
            List[str]: List of generated outputs
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        pass
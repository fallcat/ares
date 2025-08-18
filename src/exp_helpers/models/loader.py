from typing import Dict, Any, Optional
from .openai_llm import OpenAILLM
from .phi_llm import PhiLLM
from .qwen_llm import QwenLLM
from .base_llm import BaseLLM
from .flan_t5_llm import FlanT5LLM
from .prm_model import PRMModel
from .llm_apis import LLMAPI
from .vllm import VLLM
from .qwen_prm_model import QwenPRMModel
from .qwen_prm_vllm import QwenPRMVLLM

def get_llm(model_type: str, model_name: str, **kwargs: Dict[str, Any]) -> BaseLLM:
    """
    Load an LLM model based on the provided model name.

    Args:
        model_name (str): The name of the model to load.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        BaseLLM: The loaded LLM model.
    """
    if model_type == "openai":
        return OpenAILLM(model_name, **kwargs)
    elif model_type == "phi":
        return PhiLLM(model_name, **kwargs)
    elif model_type == "qwen":
        return QwenLLM(model_name, **kwargs)
    elif model_type == "flan-t5":
        return FlanT5LLM(model_name, **kwargs)
    elif model_type == "prm":
        return PRMModel(model_name, **kwargs)
    elif model_type == "llm_api":
        return LLMAPI(model_name, **kwargs)
    elif model_type == "vllm":
        return VLLM(model_name, **kwargs)
    elif model_type == "qwen_prm":
        return QwenPRMModel(model_name, **kwargs)
    elif model_type == "qwen_prm_vllm":
        return QwenPRMVLLM(model_name, **kwargs)
    else:
        raise ValueError(f"Model type {model_type} not supported")

import os
import torch
import transformers
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from .prm_model import PRMModel
import torch.nn.functional as F


def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

class QwenPRMModel(PRMModel):
    """
    Implementation for Qwen PRM models using Hugging Face.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
                 cache_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Qwen PRM LLM.
        
        Args:
            model_name (str): Name of the Qwen model to use
            cache_dir (str, optional): Directory to cache models
            **kwargs: Additional model parameters
        """
        
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
        self.model = transformers.AutoModel.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        self.step_tag = '<extra_0>'
        self.step_sep_id = self.tokenizer.encode("<extra_0>")[0]

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

    def format_prompt(self, premises, hypotheses):
        """
        premises:   list of list of str
        hypotheses: list of list of str
        """
        prompts = []
        messages_all = []
        for premises_i, hypotheses_i in zip(premises, hypotheses):
            messages = [
                {"role": "system", "content": "Please reason step-by-step."},
                {"role": "user", "content": ' '.join(premises_i)},
                {"role": "assistant", "content": "<extra_0>".join(hypotheses_i) + "<extra_0>"},
            ]
            messages_all.append(messages)
        prompts = self.tokenizer.apply_chat_template(
            messages_all, 
            tokenize=False, 
            add_generation_prompt=False,
        )
        print(prompts)
        return prompts
        
    def generate(self, 
                prompts: List[str], 
                temperature: float = 0.0,
                **kwargs) -> List[str]:
        """
        Generate text for a list of prompts using Qwen PRM model.
        
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
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Computing step rewards", unit="batch"):
            batch = prompts[i:i+batch_size]

            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, truncation=False
            ).to(self.model.device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model_outputs = self.model(**inputs)

            input_ids = inputs['input_ids']
            token_masks = (input_ids == self.step_sep_id)
            step_scores_list = make_step_rewards(model_outputs[0], token_masks)
            
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
            "model_type": "qwen_prm",
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
            "Qwen/Qwen2.5-Math-PRM-7B",
            "Qwen/Qwen2.5-Math-PRM-32B",
        ] 
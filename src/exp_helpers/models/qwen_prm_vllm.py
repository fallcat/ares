import os
import torch
import transformers
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from .prm_model import PRMModel
import torch.nn.functional as F
from transformers import AutoTokenizer


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

class QwenPRMVLLM(PRMModel):
    """
    Implementation for Qwen PRM models using vLLM.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
                 cache_dir: Optional[str] = None,
                 max_model_len: int = 4096,
                 use_tqdm: bool = False,
                 **kwargs):
        """
        Initialize the Qwen PRM vLLM.
        
        Args:
            model_name (str): Name of the Qwen model to use
            cache_dir (str, optional): Directory to cache models
            max_model_len (int): Maximum model length for vLLM
            use_tqdm (bool): Whether to use tqdm for progress
            **kwargs: Additional model parameters
        """
        
        # Set Hugging Face cache directory
        if cache_dir:
            os.environ['HF_HOME'] = cache_dir
        
        # Initialize tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set padding configuration
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        # Set default parameters
        self.default_params = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_new_tokens": kwargs.get("max_new_tokens", 500),
            "top_p": kwargs.get("top_p", 0.8),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1)
        }
        
        # Set step-related tokens
        self.step_tag = '<extra_0>'
        self.step_sep_id = self.tokenizer.encode("<extra_0>")[0]
        
        # Initialize vLLM model with reward task
        print(f"Initializing vLLM with PRM model {model_name}")
        
        # Initialize vLLM model with reward task
        # Note: override_pooler_config might be passed as a string in CLI format
        # or the pooling config might be automatically set for reward models
        self.model = LLM(
            model=model_name,
            tokenizer=model_name,
            tensor_parallel_size=1,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,  # PRM models often need eager mode
            task="reward",  # Use reward task for PRM models
            # The model should automatically configure STEP pooling for PRM models
        )
        print("vLLM initialized with reward task.")
        
        self.use_tqdm = use_tqdm
        self.model_name = model_name

    def format_prompt(self, premises, hypotheses):
        """
        premises:   list of list of str
        hypotheses: list of list of str
        """
        if len(premises) != len(hypotheses):
            raise ValueError(f"Number of premises and hypotheses must be the same. Got {len(premises)} premises and {len(hypotheses)} hypotheses.")
        
        if isinstance(hypotheses, list) and not isinstance(hypotheses[0], list):
            hypotheses = [[hypotheses[i]] for i in range(len(hypotheses))]

        prompts = []
        messages_all = []
        self.expected_steps_per_sample = []  # Track expected number of steps
        for premises_i, hypotheses_i in zip(premises, hypotheses):
            messages = [
                {"role": "system", "content": "Please reason step-by-step."},
                {"role": "user", "content": ' '.join(premises_i)},
                {"role": "assistant", "content": "<extra_0>".join(hypotheses_i) + "<extra_0>"},
            ]
            messages_all.append(messages)
            # Each hypothesis gets one step score
            self.expected_steps_per_sample.append(len(hypotheses_i))
        prompts = self.tokenizer.apply_chat_template(
            messages_all, 
            tokenize=False, 
            add_generation_prompt=False,
        )
        # print(prompts)
        return prompts
        
    def generate(self, 
                prompts: List[str], 
                temperature: float = 0.0,
                **kwargs) -> List[str]:
        """
        Generate text for a list of prompts using Qwen PRM model.
        
        Args:
            prompts (List[str]): List of input prompts
            temperature (float): Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List[str]: List of generated outputs (step scores)
        """
        return self.batch_generate(prompts, batch_size=len(prompts), 
                                 temperature=temperature, 
                                 **kwargs)
    
    def batch_generate(self, 
                      prompts: List[str], 
                      batch_size: int = 8,
                      **kwargs) -> List[str]:
        """
        Generate step scores for a list of prompts in batches using vLLM.
        
        Args:
            prompts (List[str]): List of input prompts
            batch_size (int): Size of each batch
            **kwargs: Additional generation parameters
            
        Returns:
            List[List[float]]: List of step scores for each prompt
        """
        outputs = []
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), batch_size), 
                     desc="Computing step rewards", 
                     unit="batch", 
                     disable=not self.use_tqdm):
            batch = prompts[i:i+batch_size]
            
            # For reward models, we use the encode method
            # vLLM will handle the step-wise scoring based on the pooler config
            vllm_outputs = self.model.encode(batch)
            
            batch_scores = []
            for output in vllm_outputs:
                # Extract the rewards/scores
                # With STEP pooling, this should give us scores for each step
                data = output.outputs.data
                
                # Convert data to list of floats
                if isinstance(data, torch.Tensor):
                    raw_scores = data.cpu().tolist()
                elif isinstance(data, list):
                    raw_scores = data
                else:
                    # Single score case
                    raw_scores = [[float(data)]]
                
                # For PRM models, the output is [negative_prob, positive_prob] for each step
                # We want to extract only the positive probability (index 1)
                step_scores = []
                for score in raw_scores:
                    if isinstance(score, list) and len(score) == 2:
                        # Extract positive probability (second value)
                        step_scores.append(score[1])
                    else:
                        # If not in expected format, use as is
                        step_scores.append(score if not isinstance(score, list) else score[0])
                
                # If we have expected steps info, truncate to match
                if hasattr(self, 'expected_steps_per_sample') and len(outputs) < len(self.expected_steps_per_sample):
                    expected_steps = self.expected_steps_per_sample[len(outputs)]
                    if len(step_scores) > expected_steps:
                        # Take only the last N scores where N is the expected number
                        # This assumes the extra scores are from the prompt/context
                        step_scores = step_scores[-expected_steps:]
                
                batch_scores.append(step_scores)
            
            outputs.extend(batch_scores)
            
        # Debug: print the shape of outputs
        if outputs and hasattr(self, 'expected_steps_per_sample'):
            # print(f"[QwenPRMVLLM] Number of samples: {len(outputs)}")
            for i, (scores, expected) in enumerate(zip(outputs[:3], self.expected_steps_per_sample[:3])):
                # print(f"  Sample {i}: {len(scores)} step scores (expected: {expected})")
                if len(scores) != expected:
                    print(f"    WARNING: Mismatch! Got {len(scores)}, expected {expected}")
                # if len(scores) > 5:
                #     print(f"    Last 5 scores: {scores[-5:]}")
                # else:
                #     print(f"    All scores: {scores}")
            
        return outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Qwen PRM model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "model_name": self.model_name,
            "model_type": "qwen_prm_vllm",
            "provider": "vllm",
            "tokenizer": self.tokenizer.__class__.__name__,
            "backend": "vllm",
            "default_params": self.default_params
        }
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get a list of available Qwen PRM models.
        
        Returns:
            List[str]: List of model names
        """
        return [
            "Qwen/Qwen2.5-Math-PRM-7B",
            "Qwen/Qwen2.5-Math-PRM-32B",
        ]
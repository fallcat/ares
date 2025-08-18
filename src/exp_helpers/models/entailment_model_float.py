import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union
import json
from .base_llm import BaseLLM

def parse_json_from_output(text: str) -> Dict[str, str]:
    """
    Parse JSON from model output text.
    
    Args:
        text (str): Raw text output from model
        
    Returns:
        Dict[str, str]: Parsed JSON object
    """
    # Find the JSON part in the text
    start_idx = text.find('{')
    end_idx = text.rfind('}') + 1
    
    if start_idx == -1 or end_idx == 0:
        raise ValueError("No JSON object found in text")
        
    json_str = text[start_idx:end_idx]
    return json.loads(json_str)

class EntailmentModel(nn.Module):
    """
    Model for evaluating entailment using any LLM that inherits from BaseLLM.
    """
    
    def __init__(self, 
                 llm: BaseLLM,
                 entailment_system_prompt: Optional[str] = None,
                 example_prompts: str = "",
                 new_example_prompt: Optional[str] = None,
                 entailment_mapping: Optional[Dict[str, int]] = None,
                 entailment_name: str = 'Entailment',
                 max_new_tokens: int = 500):
        """
        Initialize the entailment model.
        
        Args:
            llm (BaseLLM): LLM model to use for entailment
            entailment_system_prompt (str, optional): System prompt for entailment
            example_prompts (str): Example prompts for few-shot learning
            new_example_prompt (str, optional): Format for new examples
            entailment_mapping (Dict[str, int], optional): Mapping from entailment labels to scores
            entailment_name (str): Name of the entailment field in output
            max_new_tokens (int): Maximum number of tokens to generate
        """
        super().__init__()
        
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        
        # Set default system prompt if not provided
        if entailment_system_prompt is None:
            entailment_system_prompt = """
You are an expert judge for evaluating entailment. Given claims/reasoning chain as evidence and a hypothesis, determine if all the claims together supports that hypothesis is correct. 
The claims support the hypothesis if given the claims are true, we know the hypothesis is true.
Please do not assume knowledge not mentioned in the context, and only use the knowledge explicitly stated in the context.
Provide your judgment as one of the following: "Very Likely", "Likely", "Somewhat Likely", "Neutral", "Somewhat Unlikely", "Unlikely", "Very Unlikely".

Input format:
Context:
<claims as evidence>

Hypothesis
<hypothesis claim>

The output format must be the following format without additional words.
```json
{
"Proof": "<proof>",
"Entailment": "<Very Likely/Likely/Somewhat Likely/Neutral/Somewhat Unlikely/Unlikely/Very Unlikely>"
}
```
###
"""
        
        # Set default example prompt format if not provided
        if new_example_prompt is None:
            new_example_prompt = """
Context:
{}

Hypothesis:
{}

Output:
"""
            
        # Set default entailment mapping if not provided
        if entailment_mapping is None:
            entailment_mapping = {
                'Very Likely': 1,
                'Likely': 0.8,
                'Somewhat Likely': 0.6,
                'Neutral': 0.5,
                'Somewhat Unlikely': 0.4,
                'Unlikely': 0.2,
                'Very Unlikely': 0
            }
            
        self.entailment_system_prompt = entailment_system_prompt
        self.example_prompts = example_prompts
        self.new_example_prompt = new_example_prompt
        self.entailment_mapping = entailment_mapping
        self.entailment_name = entailment_name
        
    def _prepare_prompts(self, 
                        premises: List[str], 
                        hypotheses: List[str], 
                        s: Optional[List[torch.Tensor]] = None) -> List[str]:
        """
        Prepare prompts for the LLM.
        
        Args:
            premises (List[str]): List of premise strings
            hypotheses (List[str]): List of hypothesis strings
            s (List[torch.Tensor], optional): List of boolean tensors for subset selection
            
        Returns:
            List[str]: List of formatted prompts
        """
        if s is None:
            s = [torch.ones(len(premises[i])).bool() for i in range(len(premises))]
            
        # Format premises based on subset selection
        premises_str = [' '.join([premise[i.item()] for i in s[pi].nonzero().view(-1)]) 
                       for pi, premise in enumerate(premises)]
        
        # Create prompts
        prompts = []
        for premise, hypothesis in zip(premises_str, hypotheses):
            prompt = self.entailment_system_prompt + self.example_prompts + \
                    self.new_example_prompt.format(premise, hypothesis)
            prompts.append(prompt)
            
        return prompts
        
    def forward(self, 
                premises: List[str], 
                hypothesis: List[str], 
                s: Optional[List[torch.Tensor]] = None,
                return_all: bool = False) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass of the entailment model.
        
        Args:
            premises (List[str]): List of premise strings
            hypothesis (List[str]): List of hypothesis strings
            s (List[torch.Tensor], optional): List of boolean tensors for subset selection
            return_all (bool): Whether to return all outputs or just scores
            
        Returns:
            Union[torch.Tensor, Dict[str, Any]]: Entailment scores or all outputs
        """
        # Prepare prompts
        prompts = self._prepare_prompts(premises, hypothesis, s)
        
        # Generate outputs
        with torch.no_grad():
            raw_outputs = self.llm.generate(
                prompts,
                max_new_tokens=self.max_new_tokens
            )
            
        # Parse outputs
        entailment_outputs = []
        for output in raw_outputs:
            try:
                parsed = parse_json_from_output(output)
                entailment_outputs.append(parsed)
            except Exception as e:
                print(f"Error parsing output: {e}")
                # import pdb; pdb.set_trace()
                entailment_outputs.append({self.entailment_name: "Very Unlikely"})
                
        # Convert to scores
        entailment_scores = torch.tensor([
            self.entailment_mapping.get(output.get(self.entailment_name, "Very Unlikely"), 0) 
            for output in entailment_outputs
        ])
        
        if return_all:
            return {
                'entailment_scores': entailment_scores,
                'entailment_outputs': entailment_outputs
            }
        else:
            return entailment_scores
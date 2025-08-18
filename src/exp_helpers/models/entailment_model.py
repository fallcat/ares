import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union, Literal
import json
from .base_llm import BaseLLM
import re
from .prm_model import PRMModel
import math
from tqdm.auto import tqdm

def parse_json_from_output(text: str) -> Dict[str, str]:
    """
    Parse a flat JSON object whose values are all strings,
    correctly handling un‑escaped double‑quotes inside those strings.
    """
    # 1) find the braces
    start = text.find('{')
    end   = text.rfind('}')
    if start < 0 or end < 0:
        raise ValueError("No JSON object found in text")

    body = text[start:end+1]

    # 2) find all "key": "value" pairs, where the closing " of value
    #    is the one followed by a comma or closing brace.
    pattern = re.compile(
        r'"(?P<key>[^"]+)"\s*:\s*"'              # opening "key": "
        r'(?P<val>.*?)(?<!\\)"'                  #  capture until an un‑escaped "
        r'(?=\s*(?:,|\}))',                      #   which is followed by , or }
        re.DOTALL
    )

    result: Dict[str, str] = {}
    for m in pattern.finditer(body):
        # m.group("val") contains the full string, including any interior quotes
        result[m.group("key")] = m.group("val")

    if not result:
        raise ValueError("Failed to parse any key/value pairs")

    return result

def parse_json_llm_judge(input_text: str) -> List[Dict[str, Any]]:
    """
    Extract and parse a JSON array of claim objects from `input_text`.
    Handles both fenced (```json … ```) and unfenced JSON.
    
    Returns:
        A list of dicts with keys 'claim_id', 'reasoning', and 'entailed'.
    """
    # 1. Try to find a fenced JSON block
    fenced = re.search(r"```json\s*(\[\s*{.*?}\s*\])\s*```", input_text, re.DOTALL)
    if fenced:
        json_str = fenced.group(1)
    else:
        # 2. Otherwise, try to grab the first [...] substring
        match = re.search(r"(\[\s*{.*?}\s*\])", input_text, re.DOTALL)
        if not match:
            # import pdb; pdb.set_trace()
            raise ValueError("No JSON array found in input")
            
        json_str = match.group(1)
    
    # Fix the LaTeX parens: escape any single \ before ( or )
    # json_str_fixed = re.sub(r'(?<!\\)\\([()])', r'\\\\\1', json_str)
    json_str_dollar = re.sub(r'\\\\\((.*?)\\\\\)', r'$\1$', json_str)
    # 3. Parse it
    try:
        data = json.loads(json_str_dollar)
    except json.JSONDecodeError as e:
        import pdb; pdb.set_trace()
        raise ValueError(f"Invalid JSON: {e}")
    
    # 4. (Optional) Validate structure
    # for obj in data:
    #     if not all(key in obj for key in ("claim_id", "reasoning", "entailed")):
    #         raise ValueError(f"Missing keys in object: {obj}")
    
    return data



class EntailmentConfig:
    """Configuration class for entailment settings."""
    def __init__(self,
                 system_prompt: str,
                 mapping: Dict[str, float],
                 field_name: str,
                 default_value: str):
        self.system_prompt = system_prompt
        self.mapping = mapping
        self.field_name = field_name
        self.default_value = default_value

class EntailmentModel(nn.Module):
    """
    Model for evaluating entailment using any LLM that inherits from BaseLLM.
    Supports binary, granular, and custom probability modes.
    """
    
    # Define system prompts as class attributes
    BINARY_SYSTEM_PROMPT = """
You are an expert judge for evaluating entailment. Given claims as evidence and a hypothesis, determine if all the claims together supports that hypothesis is correct. 
The claims support the hypothesis if given the claims are true, we know the hypothesis is true.
Please do not assume knowledge not mentioned in the context, and only use the knowledge explicitly stated in the context.
Provide your judgment as one of the following: "YES", or "NO".

Input format:
Context:
<claims as evidence>

Hypothesis
<hypothesis claim>

The output format must be the following format without additional words.
```json
{
"Proof": "<proof>",
"Entail": "<YES/NO>"
}
```
###
"""

    GRANULAR_SYSTEM_PROMPT = """
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

    # Define mappings as class attributes
    BINARY_MAPPING = {
        'YES': 1,
        'NO': 0
    }

    GRANULAR_MAPPING = {
        'Very Likely': 1.0,
        'Likely': 0.8,
        'Somewhat Likely': 0.6,
        'Neutral': 0.5,
        'Somewhat Unlikely': 0.4,
        'Unlikely': 0.2,
        'Very Unlikely': 0.0
    }

    # Define mode configurations
    MODE_CONFIGS = {
        'binary': EntailmentConfig(
            system_prompt=BINARY_SYSTEM_PROMPT,
            mapping=BINARY_MAPPING,
            field_name='Entail',
            default_value='NO'
        ),
        'granular': EntailmentConfig(
            system_prompt=GRANULAR_SYSTEM_PROMPT,
            mapping=GRANULAR_MAPPING,
            field_name='Entailment',
            default_value='Neutral'
        )
    }
    
    def __init__(self, 
                 llm: BaseLLM,
                 example_prompts: str = "",
                 new_example_prompt: Optional[str] = None,
                 max_new_tokens: int = 500,
                 batch_size: int = 8,
                 verbose: bool = False,
                 claim_delimiter: str = " ",
                 debug: bool = False):
        """
        Initialize the entailment model.
        
        Args:
            llm (BaseLLM): LLM model to use for entailment
            example_prompts (str): Example prompts for few-shot learning
            new_example_prompt (str, optional): Format for new examples
            max_new_tokens (int): Maximum number of tokens to generate
        """
        super().__init__()
        
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        
        # Set default example prompt format if not provided
        if new_example_prompt is None:
            new_example_prompt = """
Context:
{}

Hypothesis:
{}

Output:
"""
        
        self.example_prompts = example_prompts
        self.new_example_prompt = new_example_prompt
        self.batch_size = batch_size
        self.verbose = verbose
        self.claim_delimiter = claim_delimiter
        self.debug = debug
    @classmethod
    def create_custom_config(cls, 
                           system_prompt: str,
                           mapping: Dict[str, float],
                           field_name: str,
                           default_value: str) -> EntailmentConfig:
        """
        Create a custom entailment configuration.
        
        Args:
            system_prompt (str): Custom system prompt
            mapping (Dict[str, float]): Custom mapping from labels to scores
            field_name (str): Name of the field in JSON output
            default_value (str): Default value when parsing fails
            
        Returns:
            EntailmentConfig: Custom configuration
        """
        return EntailmentConfig(
            system_prompt=system_prompt,
            mapping=mapping,
            field_name=field_name,
            default_value=default_value
        )
        
    def _prepare_prompts(self, 
                        premises: List[str], 
                        hypotheses: List[str],
                        config: EntailmentConfig,
                        s: Optional[List[torch.Tensor]] = None,
                        num_raw: int = 0) -> List[str]:
        """
        Prepare prompts for the LLM.
        
        Args:
            premises (List[str]): List of premise strings
            hypotheses (List[str]): List of hypothesis strings
            config (EntailmentConfig): Entailment configuration
            s (List[torch.Tensor], optional): List of boolean tensors for subset selection
            
        Returns:
            List[str]: List of formatted prompts
        """
        if s is None:
            s = [torch.ones(len(premises[i])).bool() for i in range(len(premises))]
            
        # Format premises based on subset selection

        if isinstance(self.llm, PRMModel): # no need to format premises for PRM models
            # check if the llm has self.format_prompt method
            if hasattr(self.llm, 'format_prompt'):
                prompts = self.llm.format_prompt(premises, hypotheses)
            else:
                premises_str = [' '.join([premise[i.item()] for i in s[pi].nonzero().view(-1)]) 
                            for pi, premise in enumerate(premises)]

                if isinstance(hypotheses[0], list):
                    hypotheses_str = [(' ' + self.llm.step_tag + '\n').join(hypothesis) + ' ' + self.llm.step_tag
                                    for hi, hypothesis in enumerate(hypotheses)]
                else:
                    hypotheses_str = [hypothesis + ' ' + self.llm.step_tag
                                    for hi, hypothesis in enumerate(hypotheses)]

                # Create prompts
                prompts = []
                for premise, hypothesis in zip(premises_str, hypotheses_str):
                    prompt = premise +  ' ' + hypothesis
                    prompts.append(prompt)
        
        else:
            premises_str = []
            hypotheses_str = []
            for pi in range(len(premises)):
                premise = premises[pi]
                hypothesis = hypotheses[pi]
                if not isinstance(premise, list):
                    premise = [premise]
                if not isinstance(hypothesis, list):
                    hypothesis = [hypothesis]

                try:
                    if num_raw > 0: # if we consider putting raw claims separately in the context.
                        if self.claim_delimiter == " ":
                            premises_str.append(json.dumps([premise[i.item()] for i in s[pi][:num_raw].nonzero().view(-1)], indent=4))
                        else:
                            premises_str.append(self.claim_delimiter.join([premise[i.item()] for i in s[pi][:num_raw].nonzero().view(-1)]))
                    else: # old way
                        premises_str.append(self.claim_delimiter.join([premise[i.item()] for i in s[pi].nonzero().view(-1)]))
                except Exception as e:
                    import pdb; pdb.set_trace()
                    # if self.claim_delimiter == " ":
                    #     premises_str.append(json.dumps([premise[i.item()] for i in s[pi][:num_raw].nonzero().view(-1)], indent=4))
                    # else:
                    #     premises_str.append(self.claim_delimiter.join([premise[i.item()] for i in s[pi][:num_raw].nonzero().view(-1)]))
                
                if num_raw > 0: # if we consider putting raw claims separately in the context.
                    # put second half of premise and also hypothesis
                    premise_but_derived = [premise[num_raw+i.item()] for i in s[pi][num_raw:].nonzero().view(-1)]
                    hypotheses_list = premise_but_derived + hypothesis
                    if self.claim_delimiter == " ":
                        hypotheses_str.append(json.dumps(hypotheses_list, indent=4))
                    else:
                        hypotheses_str.append(self.claim_delimiter.join(hypotheses_list))
                else:
                    hypotheses_str.append(self.claim_delimiter.join(hypothesis))
            
            # Create prompts
            prompts = []
            for premise, hypothesis in zip(premises_str, hypotheses_str):
                prompt = config.system_prompt + self.example_prompts + \
                        self.new_example_prompt.format(premise, hypothesis)
                prompts.append(prompt)
            
        return prompts
        
    def forward(self, 
                premises: List[str], 
                hypothesis: List[str], 
                s: Optional[List[torch.Tensor]] = None,
                return_all: bool = False,
                temperature: Optional[float] = 0.0,
                mode: Union[Literal['binary', 'granular'], EntailmentConfig] = 'binary',
                num_raw: int = 0
        ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass of the entailment model.
        
        Args:
            premises (List[str]): List of premise strings
            hypothesis (List[str]): List of hypothesis strings
            mode (Union[str, EntailmentConfig]): 'binary', 'granular', or custom EntailmentConfig
            s (List[torch.Tensor], optional): List of boolean tensors for subset selection
            return_all (bool): Whether to return all outputs or just scores
            temperature (float, optional): Temperature for generation
            
        Returns:
            Union[torch.Tensor, Dict[str, Any]]: Entailment scores or all outputs
        """
        # Get configuration based on mode
        if isinstance(mode, str):
            if mode not in self.MODE_CONFIGS:
                raise ValueError(f"Invalid mode: {mode}. Must be 'binary' or 'granular'")
            config = self.MODE_CONFIGS[mode]
        else:
            config = mode
        
        # Prepare prompts
        prompts = self._prepare_prompts(premises, hypothesis, config, s, num_raw)
        
        # Generate outputs
        with torch.no_grad():
            raw_outputs = []
            pbar = range(0, len(prompts), self.batch_size)
            
            if self.verbose:
                pbar = tqdm(pbar)
            for i in pbar:
                raw_outputs.extend(self.llm.generate(
                    prompts[i:i+self.batch_size],
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature
                ))
            
        if self.debug:
            print('raw_outputs', raw_outputs)
        # import pdb; pdb.set_trace()
        # Parse outputs
        entailment_scores_all = []
        if isinstance(raw_outputs[0], str):
            if num_raw > 0:
                entailment_outputs = []
                entailment_scores = []
                entailment_scores_all = []
                for oi, output in enumerate(raw_outputs):
                    try:
                        parsed = parse_json_llm_judge(output)
                        entailment_outputs.append(parsed)
                    except Exception as e:
                        # import pdb; pdb.set_trace()
                        # parse as raw text and by line, each line has format claim_id, entailment_name
                        lines = output.split('\n')
                        parsed = []
                        for line in lines:
                            if line.strip():
                                try:
                                    claim_id, entailment_name = line.split(':')[0].strip(), line.split(':')[1].strip()
                                    # delete all the asterisks and new lines
                                    entailment_name = re.sub(r'[*]', '', entailment_name)
                                    if self.debug:
                                        print('')
                                        print('claim_id', claim_id, 'entailment_name', entailment_name)
                                    parsed.append({'claim_id': claim_id, config.field_name: entailment_name})
                                    # entailment_outputs.append([config.mapping[entailment_name.strip()]])
                                except Exception as e:
                                    if len(parsed) >= len(premises[oi]) - num_raw:
                                        continue
                                    import pdb; pdb.set_trace()
                                    # if we cannot parse the line, use default value
                                    parsed.append({'claim_id': claim_id, config.field_name: config.default_value})
                        entailment_outputs.append(parsed)
                                
                    print("entailment_outputs", entailment_outputs)
                    print("parsed", parsed)
                    try:
                        entailment_scores_i = torch.tensor([config.mapping[output[config.field_name].replace("<", "").replace(">", "")] for output in parsed if output[config.field_name] in config.mapping])
                    except:
                        # use default value
                        entailment_scores_i = torch.tensor([config.mapping[config.default_value]])
                        print("failed parsing, using default value")
                    # import pdb; pdb.set_trace()
                    print("entailment_scores_i", entailment_scores_i)
                    entailment_scores.append(entailment_scores_i[-1])
                    entailment_scores_all.append(entailment_scores_i)
                    # import pdb; pdb.set_trace()
                    # entailment_scores_i = torch.tensor([config.mapping[output[config.field_name]] for output in parsed])
                entailment_scores = torch.stack(entailment_scores)
            else: # previous way
                entailment_outputs = []
                for output in raw_outputs:
                    try:
                        # Case 1: Try to parse as JSON first
                        parsed = parse_json_from_output(output)
                        entailment_outputs.append(parsed)
                    except Exception as e:

                        
                        
                        try:
                            # Case 2: If JSON parsing fails, try to use the raw text directly
                            stripped_output = output.strip()
                            if config.field_name + ':' in stripped_output:
                                # get the value after the colon
                                entailment_name = stripped_output.split(f'{config.field_name}:')[1].strip()
                                # entailment_name = entailment_name.replace('*', '')
                                # entailment_name = entailment_name.split('\n')[0].strip()
                                
                                # Sort mapping keys by length (longest first) and find matching key
                                sorted_keys = sorted(config.mapping.keys(), key=len, reverse=True)
                                for key in sorted_keys:
                                    if key.lower() in entailment_name.lower():
                                        entailment_name = key
                                        break
                                        
                                if self.debug:
                                    print('')
                                    print('entailment_name', entailment_name)
                                    print()
                                entailment_outputs.append({config.field_name: entailment_name})
                                # print('parsed', {config.field_name: stripped_output.split(f'{config.field_name}:')[1].strip()})
                            else:
                                # Check if the stripped output is directly in the mapping
                                if stripped_output in config.mapping:
                                    entailment_outputs.append({config.field_name: stripped_output})
                                else:
                                    # Fall back to default value if stripped output isn't in mapping
                                    entailment_outputs.append({config.field_name: config.default_value})
                        except Exception as e2:
                            print(f"Error processing raw output: {e2}")
                            entailment_outputs.append({config.field_name: config.default_value})

                # Convert to scores
                entailment_scores = torch.tensor([
                    config.mapping.get(output.get(config.field_name, config.default_value), 0.0) 
                    for output in entailment_outputs
                ])
        else:
            entailment_scores = torch.tensor([raw_output[-1] for raw_output in raw_outputs])
            entailment_outputs = raw_outputs

        
        if return_all:
            return {
                'entailment_scores': entailment_scores,
                'entailment_outputs': entailment_outputs,
                'mode': 'custom' if isinstance(mode, EntailmentConfig) else mode,
                'config': config.__dict__,
                'entailment_scores_all': entailment_scores_all,
                'prompts': prompts
            }
        else:
            return entailment_scores
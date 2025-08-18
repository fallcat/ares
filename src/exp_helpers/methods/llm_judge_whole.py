from .base_scorer import BaseStabilityScorer, StabilityRateResults
from ..models.base_llm import BaseLLM
from typing import Dict, Any, List, Union
import json
import re


def parse_any_json_from_output(output: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    try:
        # Extract content between ```json and ``` markers
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, output)
        if match:
            json_content = match.group(1)
            try:
                return json.loads(json_content)
            except json.JSONDecodeError:
                return None
                # import pdb; pdb.set_trace()
                # raise ValueError(f"Failed to parse JSON from output: {output}")
        else:
            # If no markers found, try parsing the whole output
            return json.loads(output)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON from output: {output}")

class LLMJudgeWholeStabilityScorer(BaseStabilityScorer):
    BINARY_SYSTEM_PROMPT = """
You are an expert judge for evaluating entailment. Given raw claims and a reasoning trace with derived claims, determine if each derived claim is correct in the reasoning.
The claims support the hypothesis if given the claims are true, we know the hypothesis is true.
Please do not assume knowledge not mentioned in the context, and only use the knowledge explicitly stated in the context.
Provide your judgment as one of the following: "YES", or "NO".

Input format:
Raw Claims:
<raw claims to start with>

Derived Claims:
<derived claims>

The output format must be the following format without additional words.
```json
{
    "raw_claims": [
        "raw_claim_1",
        "raw_claim_2",
        ...
    ],
    "derived_claims": [
        {
            "claim": "derived_claim_1",
            "reasoning": "reasoning_1",
            "entailed": "<YES/NO>"
        },
        ...
    ]
}
```
###
"""

    GRANULAR_SYSTEM_PROMPT = """
You are an expert judge for evaluating entailment. Given raw claims and a reasoning trace with derived claims, determine if each derived claim is correct in the reasoning.
The claims support the hypothesis if given the claims are true, we know the hypothesis is true.
Please do not assume knowledge not mentioned in the context, and only use the knowledge explicitly stated in the context.
Provide your judgment as one of the following: "Very Likely", "Likely", "Somewhat Likely", "Neutral", "Somewhat Unlikely", "Unlikely", "Very Unlikely".

Input format:
Raw Claims:
<raw claims to start with>

Derived Claims:
<derived claims>

The output format must be the following format without additional words.
```json
{
    "raw_claims": [
        "raw_claim_1",
        "raw_claim_2",
        ...
    ],
    "derived_claims": [
        {
            "claim": "derived_claim_1",
            "reasoning": "reasoning_1",
            "entailed": "<Very Likely/Likely/Somewhat Likely/Neutral/Somewhat Unlikely/Unlikely/Very Unlikely>"
        },
        ...
    ]
}
```
###
"""

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

    def __init__(self, llm: BaseLLM, p: float = 0.95,
                 epsilon: float = 0.1, delta: float = 0.1, 
                 system_prompt: str = None,
                 max_tokens: int = 2000,
                 entailment_mode: str = 'binary',
                 mapping: Dict[str, float] = None):
        if system_prompt is None:
            if entailment_mode == 'binary':
                system_prompt = self.BINARY_SYSTEM_PROMPT
            else:
                system_prompt = self.GRANULAR_SYSTEM_PROMPT
        self.llm = llm
        self.p = p
        self.epsilon = epsilon
        self.delta = delta
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.entailment_mode = entailment_mode
        if mapping is None:
            if entailment_mode == 'binary':
                self.mapping = self.BINARY_MAPPING
            else:
                self.mapping = self.GRANULAR_MAPPING
        else:
            self.mapping = mapping

    def get_stability_rate(self, input_dict: Dict[str, Any]) -> StabilityRateResults:

        raw_claims = input_dict['ent_inputs'][0]['premises']
        derived_claims = [input_dict['ent_inputs'][i]['hypothesis'] for i in range(len(input_dict['ent_inputs']))]
        
        prompt = f"""
Raw Claims:
{json.dumps(raw_claims, indent=4)}

Derived Claims:
{json.dumps(derived_claims, indent=4)}
"""
        response = self.llm.generate([self.system_prompt + prompt], max_new_tokens=self.max_tokens)
        parsed_response = parse_any_json_from_output(response[0])
        if parsed_response is None:
            print(f"Failed to parse JSON from output: {response[0]}")
            if self.entailment_mode == 'binary':
                stability_rates = [0] * len(derived_claims)
                stability_rate = 0
            else:
                stability_rates = [0.5] * len(derived_claims)
                stability_rate = 0.5
        else:
            try:
                try:
                    stability_rates = [self.mapping[claim['entailed']] if 'entailed' in claim and claim['entailed'] in self.mapping 
                                    else (0.5 if self.entailment_mode == 'granular' else 0) 
                                    for claim in parsed_response['derived_claims']]
                except:
                     stability_rates = [self.mapping[claim['entailed']] if 'entailed' in claim and claim['entailed'] in self.mapping 
                                    else (0.5 if self.entailment_mode == 'granular' else 0) 
                                    for claim in parsed_response]
                stability_rate = stability_rates[-1]
            except KeyError:
                print(f"KeyError: {parsed_response['derived_claims']}")
                import pdb; pdb.set_trace()
                if self.entailment_mode == 'binary':
                    stability_rates = [0] * len(derived_claims)
                    stability_rate = 0
                else:
                    stability_rates = [0.5] * len(derived_claims)
                    stability_rate = 0.5

        return StabilityRateResults(
            stability_rate=stability_rate,
            stability_rates=stability_rates,
            inputs=input_dict['ent_inputs'],
            children=input_dict['children'],
            parents=input_dict['parents'],
            stab_rate_results={
                'raw_claims': raw_claims,
                'derived_claims': derived_claims,
                'response': response,
                'parsed_response': parsed_response
            }
        )

from .base_scorer import BaseStabilityScorer, StabilityRateResults
from ..models.entailment_model import EntailmentModel
from typing import Dict, Any

class RecevalIntraCorrectStabilityScorer(BaseStabilityScorer):
    def __init__(self, entailment_model: EntailmentModel, p: float = 0.95,
                 epsilon: float = 0.1, delta: float = 0.1, entailment_mode: str = 'binary',
                 split_raw_derived: bool = False, temperature: float = 0.0, rcu_prompt: str = None):
        self.entailment_model = entailment_model
        self.p = p
        self.epsilon = epsilon
        self.delta = delta
        self.entailment_mode = entailment_mode
        self.split_raw_derived = split_raw_derived
        self.temperature = temperature
        if rcu_prompt is None:
            self.rcu_prompt = """
Extract premises and conclusion from the text below. 
Please ensure that the premises and conclusions together cover all the claims in the text.

OUTPUT FORMAT:
Premises:
<premise1>
<premise2>

Hypothesis:
<hypothesis>

Input:
{}

Output:
"""
        else:
            self.rcu_prompt = rcu_prompt

    def get_stability_rate(self, input_dict: Dict[str, Any]) -> StabilityRateResults:

        num_raw = 0
        # Get entailment scores for each premise-hypothesis pair in the first input
        entailment_scores = []
        premises_list = []
        hypothesis_list = []
        for ent_input in input_dict['ent_inputs']:
            
            # Get the premises and hypothesis from the hypothesis
            claim_split = self.entailment_model.llm.generate([self.rcu_prompt.format(ent_input['hypothesis'])])
            claim_split = claim_split[0].split("Hypothesis:")
            claim_premises = [item.strip() for item in claim_split[0].split("Premises:")[1].strip().split('\n')]

            try:
                claim_hypothesis = claim_split[1].strip()
            except Exception as e:
                claim_hypothesis = claim_premises[-1] # if the step is too long and the model fails to generate the hypothesis, we use the last premise as the hypothesis
                claim_premises = claim_premises[:-1]
            premises_list.append(claim_premises)
            hypothesis_list.append(claim_hypothesis)
        
        # Get entailment scores within each hypothesis claim
        entailment_results = self.entailment_model(
            premises_list, 
            hypothesis_list, 
            mode=self.entailment_mode, 
            num_raw=num_raw,
            return_all=True,
            temperature=self.temperature
        )
        entailment_scores = entailment_results['entailment_scores'].tolist()
        entailment_outputs = entailment_results['entailment_outputs']
        entailment_scores_all = [item.tolist() for item in entailment_results['entailment_scores_all']]
        entailment_config = entailment_results['config']
        entailment_config['num_raw'] = num_raw
        entailment_prompts = entailment_results['prompts']

        stability_rate = entailment_scores[-1]
        stability_rates = entailment_scores

        return StabilityRateResults(
            stability_rate=stability_rate,
            stability_rates=stability_rates,
            inputs=input_dict['ent_inputs'],
            children=input_dict['children'],
            parents=input_dict['parents'],
            stab_rate_results={
                'entailment_scores_all': entailment_scores_all,
                'entailment_config': entailment_config,
                'entailment_outputs': entailment_outputs,
                'entailment_prompts': entailment_prompts
            }
        )

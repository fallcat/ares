from .base_scorer import BaseStabilityScorer, StabilityRateResults
from ..models.entailment_model import EntailmentModel
from typing import Dict, Any

class EntailRawStabilityScorer(BaseStabilityScorer):
    """
    This scorer is used to compute the stability of an entailment model on a claim give only raw claims.
    """
    def __init__(self, entailment_model: EntailmentModel, p: float = 0.95,
                 epsilon: float = 0.1, delta: float = 0.1, entailment_mode: str = 'binary',
                 split_raw_derived: bool = False, temperature: float = 0.0):
        self.entailment_model = entailment_model
        self.p = p
        self.epsilon = epsilon
        self.delta = delta
        self.entailment_mode = entailment_mode
        self.split_raw_derived = split_raw_derived
        self.temperature = temperature
    def get_stability_rate(self, input_dict: Dict[str, Any]) -> StabilityRateResults:
        # Get entailment scores for each premise-hypothesis pair
        if self.split_raw_derived:
            num_raw = len(input_dict['ent_inputs'][0]['premises'])
        else:
            num_raw = 0
        entailment_scores = []
        premises_list = []
        hypothesis_list = []
        for ent_input in input_dict['ent_inputs']:
            # premises = ent_input['premises']
            premises = input_dict['ent_inputs'][0]['premises'] # we will always use the first premise which is the raw claim
            hypothesis = ent_input['hypothesis']
            premises_list.append(premises)
            hypothesis_list.append(hypothesis)
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

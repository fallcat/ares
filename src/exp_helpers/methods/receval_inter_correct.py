from .base_scorer import BaseStabilityScorer, StabilityRateResults
from ..models.entailment_model import EntailmentModel
from typing import Dict, Any

class RecevalInterCorrectStabilityScorer(BaseStabilityScorer):
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
        if self.split_raw_derived:
            num_raw = len(input_dict['ent_inputs'][0]['premises'])
        else:
            num_raw = 0
        # Get entailment scores for each premise-hypothesis pair
        entailment_scores = []
        entailment_results_all = []
        
        for ent_input in input_dict['ent_inputs']:
            premises_list = []
            hypothesis_list = []
            for premise in ent_input['premises']:
                premises_list.append([premise])
                hypothesis_list.append(ent_input['hypothesis'])

            entailment_results = self.entailment_model(
                premises_list, 
                hypothesis_list, 
                mode=self.entailment_mode, 
                num_raw=num_raw,
                return_all=True,
                temperature=self.temperature
            )
            entailment_scores.append(1 - max([1 - item for item in entailment_results['entailment_scores'].tolist()])) # 1 - max contradiction score
            entailment_results_all.append(entailment_results)
        # Get the stability rate for the last input
        stability_rate = entailment_scores[-1]
        stability_rates = entailment_scores

        entailment_scores_all = [[item.tolist() for item in results['entailment_scores_all']] for results in entailment_results_all]
        entailment_config = [results['config'] for results in entailment_results_all]
        for config in entailment_config:
            config['num_raw'] = num_raw
        entailment_prompts = [results['prompts'] for results in entailment_results_all]

        # Return the results
        return StabilityRateResults(
            stability_rate=stability_rate,
            stability_rates=stability_rates,
            inputs=input_dict['ent_inputs'],
            children=input_dict['children'],
            parents=input_dict['parents'],
            stab_rate_results={
                'entailment_scores_all': entailment_scores_all,
                'entailment_config': entailment_config,
                'entailment_prompts': entailment_prompts
            }
        )

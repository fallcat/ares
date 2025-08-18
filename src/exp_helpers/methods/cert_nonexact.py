from .base_scorer import BaseStabilityScorer, StabilityRateResults
from ..models.entailment_model import EntailmentModel
from typing import Dict, Any
from .utils.stability_deterministic import tree_stability_rate_deterministic

"""
TODO: Format raw and derived into ent_inputs so that the user doesn't have to do it.
"""

class CertNonexactStabilityScorer(BaseStabilityScorer):
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
        stab_rate_results_all_hyps = tree_stability_rate_deterministic(
            self.entailment_model, 
            input_dict['ent_inputs'], 
            children=input_dict['children'],
            parents=input_dict['parents'],
            p= self.p, 
            epsilon=self.epsilon,
            delta=self.delta,
            exact=False,
            entailment_mode=self.entailment_mode,
            num_raw=num_raw,
            temperature=self.temperature
            )

        # Get the entailment rate for the hypothesis
        stability_rate = stab_rate_results_all_hyps[-1]['stab_rate_results']['stability_rate']
        stability_rates = [result['stab_rate_results']['stability_rate'] for result in stab_rate_results_all_hyps]
        return StabilityRateResults(
            stability_rate=stability_rate,
            stability_rates=stability_rates,
            inputs=input_dict['ent_inputs'],
            children=input_dict['children'],
            parents=input_dict['parents'],
            stab_rate_results=stab_rate_results_all_hyps
        )

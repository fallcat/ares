from .base_scorer import BaseStabilityScorer, StabilityRateResults
from ..models.entailment_model import EntailmentModel
from typing import Dict, Any
from .utils.stability_deterministic_equal import tree_stability_rate_deterministic_equal

class CertNonexactEqualStabilityScorer(BaseStabilityScorer):
    def __init__(self, entailment_model: EntailmentModel, p: float = 0.95,
                 epsilon: float = 0.1, delta: float = 0.1, entailment_mode: str = 'binary'):
        self.entailment_model = entailment_model
        self.p = p
        self.epsilon = epsilon
        self.delta = delta
        self.entailment_mode = entailment_mode

    def get_stability_rate(self, input_dict: Dict[str, Any]) -> StabilityRateResults:
        stab_rate_results_all_hyps = tree_stability_rate_deterministic_equal(
            self.entailment_model, 
            input_dict['ent_inputs'], 
            children=input_dict['children'],
            parents=input_dict['parents'],
            p= self.p, 
            epsilon=self.epsilon,
            delta=self.delta,
            exact=False,
            entailment_mode=self.entailment_mode,
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

from .base_scorer import BaseStabilityScorer, StabilityRateResults
from ..models.entailment_model import EntailmentModel
from typing import Dict, Any
from .utils.stability_sample import tree_stability_rate_sample

class CertSampleStabilityScorer(BaseStabilityScorer):
    def __init__(self, entailment_model: EntailmentModel, p: float = 0.95,
                 epsilon: float = 0.1, delta: float = 0.1, entailment_mode: str = 'binary', temperature: float = 0.5,
                 threshold: float = 0.6, split_raw_derived: bool = False):
        self.entailment_model = entailment_model
        self.p = p
        self.epsilon = epsilon
        self.delta = delta
        self.entailment_mode = entailment_mode
        self.temperature = temperature
        self.threshold = threshold
        self.split_raw_derived = split_raw_derived
    def get_stability_rate(self, input_dict: Dict[str, Any]) -> StabilityRateResults:
        stab_rate_results_all_hyps = tree_stability_rate_sample(
            self.entailment_model, 
            input_dict['ent_inputs'], 
            children=input_dict['children'],
            parents=input_dict['parents'],
            p= self.p, 
            epsilon=self.epsilon,
            delta=self.delta,
            entailment_mode=self.entailment_mode,
            temperature=self.temperature,
            threshold=self.threshold,
            split_raw_derived=self.split_raw_derived
            )

        # Get the entailment rate for the hypothesis
        stability_rate = stab_rate_results_all_hyps['stability_rates'][-1]
        stability_rates = stab_rate_results_all_hyps['stability_rates']

        return StabilityRateResults(
            stability_rate=stability_rate,
            stability_rates=stability_rates,
            inputs=input_dict['ent_inputs'],
            children=input_dict['children'],
            parents=input_dict['parents'],
            stab_rate_results=stab_rate_results_all_hyps
        )

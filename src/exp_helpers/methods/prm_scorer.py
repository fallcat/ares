from .base_scorer import BaseStabilityScorer, StabilityRateResults
from ..models.entailment_model import EntailmentModel
from typing import Dict, Any

class PRMStabilityScorer(BaseStabilityScorer):
    """
    This scorer is used to compute Process-level Reward Models (PRMs).
    It should be used with a model that can take in a list of premises and a hypothesis and return a score for each premise and also a score for the hypothesis.
    It should only take entailment model that are PRM models.
    """
    def __init__(self, entailment_model: EntailmentModel, entailment_mode: str = 'binary', **kwargs):
        self.entailment_model = entailment_model
        self.entailment_mode = entailment_mode

    def get_stability_rate(self, input_dict: Dict[str, Any]) -> StabilityRateResults:
        # get all the scores directly
        entailment_outputs_all = self.entailment_model(
            [input_dict['ent_inputs'][0]['premises']], 
            [[input_dict['ent_inputs'][i]['hypothesis'] for i in range(len(input_dict['ent_inputs']))]],
            mode=self.entailment_mode,
            return_all=True
            )

        entailment_scores = entailment_outputs_all['entailment_outputs'][0]

        stability_rate = entailment_scores[-1]
        stability_rates = entailment_scores

        return StabilityRateResults(
            stability_rate=stability_rate,
            stability_rates=stability_rates,
            inputs=input_dict['ent_inputs'],
            children=input_dict['children'],
            parents=input_dict['parents'],
            stab_rate_results=[]
        )

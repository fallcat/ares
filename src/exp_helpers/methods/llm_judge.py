from .base_scorer import BaseStabilityScorer, StabilityRateResults
from ..models.entailment_model import EntailmentModel
from typing import Dict, Any

class LLMJudgeStabilityScorer(BaseStabilityScorer):
    def __init__(self, entailment_model: EntailmentModel, p: float = 0.95,
                 epsilon: float = 0.1, delta: float = 0.1, entailment_mode: str = 'binary', temperature: float = 0.0):
        self.entailment_model = entailment_model
        self.p = p
        self.epsilon = epsilon
        self.delta = delta
        self.entailment_mode = entailment_mode
        self.temperature = temperature
        self.split_raw_derived = True

    def get_stability_rate(self, input_dict: Dict[str, Any]) -> StabilityRateResults:
        if self.split_raw_derived:
            num_raw = len(input_dict['ent_inputs'][0]['premises'])
        else:
            num_raw = 0
        # Get entailment scores for each premise-hypothesis pair
        # entailment_scores = []
        # premises_list = []
        # hypothesis_list = []
        # for ent_input in input_dict['ent_inputs']:
        #     premises = ent_input['premises']
        #     hypothesis = ent_input['hypothesis']
        #     premises_list.append(premises)
        #     hypothesis_list.append(hypothesis)
        # scores = self.entailment_model(
        #     premises_list, 
        #     hypothesis_list, 
        #     mode=self.entailment_mode, 
        #     num_raw=num_raw
        # )
        # # entailment_scores.append(scores.tolist())
        # entailment_scores = scores.tolist()
        # stability_rate = entailment_scores[-1]
        # stability_rates = entailment_scores

        # # import pdb; pdb.set_trace()

        # return StabilityRateResults(
        #     stability_rate=stability_rate,
        #     stability_rates=stability_rates,
        #     inputs=input_dict['ent_inputs'],
        #     children=input_dict['children'],
        #     parents=input_dict['parents'],
        #     stab_rate_results=[]
        # )

        entailment_results = self.entailment_model(
            [input_dict['ent_inputs'][-1]['premises']], 
            [input_dict['ent_inputs'][-1]['hypothesis']], 
            mode=self.entailment_mode, 
            num_raw=num_raw,
            return_all=True,
            temperature=self.temperature
        )
        # import pdb; pdb.set_trace()
        entailment_scores = entailment_results['entailment_scores_all'][0].tolist()

        # entailment_scores = entailment_results['entailment_scores'].tolist()
        entailment_outputs = entailment_results['entailment_outputs']
        entailment_scores_all = [item.tolist() for item in entailment_results['entailment_scores_all']]
        entailment_config = entailment_results['config']
        entailment_config['num_raw'] = num_raw
        entailment_prompts = entailment_results['prompts']

        # stability_rate = entailment_scores[-1]
        stability_rates = entailment_scores
        # import pdb; pdb.set_trace()

        stability_rate = entailment_scores[-1]

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

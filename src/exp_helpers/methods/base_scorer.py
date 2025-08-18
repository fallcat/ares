from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from collections import namedtuple


StabilityRateResults = namedtuple('StabilityRateResults', [
    'stability_rate',
    'stability_rates',
    'inputs',
    'children',
    'parents',
    'stab_rate_results'
])

class BaseStabilityScorer(ABC):
    @abstractmethod
    def get_stability_rate(self, input_dict: Dict[str, Any]) -> StabilityRateResults:
        pass

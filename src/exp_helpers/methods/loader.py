from .cert_exact import CertExactStabilityScorer
from .cert_nonexact import CertNonexactStabilityScorer
from .cert_sample import CertSampleStabilityScorer
from .base_scorer import BaseStabilityScorer
from .entail import EntailStabilityScorer
from .entail_raw import EntailRawStabilityScorer
from .cert_nonexact_equal import CertNonexactEqualStabilityScorer
from .llm_judge_whole import LLMJudgeWholeStabilityScorer
from .llm_judge import LLMJudgeStabilityScorer
from .receval_inter_correct import RecevalInterCorrectStabilityScorer
from .receval_intra_correct import RecevalIntraCorrectStabilityScorer
from .roscoe_li_source import RoscoeLiSourceStabilityScorer
from .roscoe_li_self import RoscoeLiSelfStabilityScorer
from ..models.entailment_model import EntailmentModel
from ..models.base_llm import BaseLLM
from .prm_scorer import PRMStabilityScorer
from typing import Union

def get_stability_scorer(
    method_name: str,
    model: Union[EntailmentModel, BaseLLM], 
    p: float = 0.95,
    **kwargs
    ) -> BaseStabilityScorer:
    
    if method_name == 'cert_exact':
        return CertExactStabilityScorer(model, p, **kwargs)
    elif method_name == 'cert_nonexact':
        return CertNonexactStabilityScorer(model, p, **kwargs)
    elif method_name == 'cert_sample':
        return CertSampleStabilityScorer(model, p, **kwargs)
    elif method_name == 'cert_nonexact_equal':
        return CertNonexactEqualStabilityScorer(model, p, **kwargs)
    elif method_name == 'entail':
        return EntailStabilityScorer(model, p, **kwargs)
    elif method_name == 'entail_raw':
        return EntailRawStabilityScorer(model, p, **kwargs)
    elif method_name == 'llm_judge_whole': # this can only work with LLM
        return LLMJudgeWholeStabilityScorer(model, p, **kwargs)
    elif method_name == 'prm': # this can only work with PRM models
        return PRMStabilityScorer(model, **kwargs)
    elif method_name == 'llm_judge':
        return LLMJudgeStabilityScorer(model, **kwargs)
    elif method_name == 'receval_inter_correct':
        return RecevalInterCorrectStabilityScorer(model, **kwargs)
    elif method_name == 'receval_intra_correct':
        return RecevalIntraCorrectStabilityScorer(model, **kwargs)
    elif method_name == 'roscoe_li_source':
        return RoscoeLiSourceStabilityScorer(model, **kwargs)
    elif method_name == 'roscoe_li_self':
        return RoscoeLiSelfStabilityScorer(model, **kwargs)
    else:
        raise ValueError(f"Method {method_name} not supported")
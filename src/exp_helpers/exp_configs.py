from .configs.data_configs.eb import DATA_CONFIGS as EB_DATA_CONFIGS
from .configs.data_configs.inli import DATA_CONFIGS as INLI_DATA_CONFIGS
from .configs.data_configs.gsm8k import DATA_CONFIGS as GSM8K_DATA_CONFIGS
from .configs.data_configs.reveal import DATA_CONFIGS as REVEAL_DATA_CONFIGS
from .configs.data_configs.gridpuzzle import DATA_CONFIGS as GRIDPUZZLE_DATA_CONFIGS
from .configs.data_configs.prmbench import DATA_CONFIGS as PRMBENCH_DATA_CONFIGS
from .configs.data_configs.deltabench import DATA_CONFIGS as DELTABENCH_DATA_CONFIGS
from .configs.data_configs.synthchain import DATA_CONFIGS as SYNTHCHAIN_DATA_CONFIGS
from .configs.data_configs.naturalsynthchain import DATA_CONFIGS as NATURALSYNTHCHAIN_DATA_CONFIGS
from .configs.data_configs.recipe_graph import DATA_CONFIGS as RECIPE_GRAPH_DATA_CONFIGS

DATA_CONFIGS = {
    **EB_DATA_CONFIGS,
    **INLI_DATA_CONFIGS,
    **GSM8K_DATA_CONFIGS,
    **REVEAL_DATA_CONFIGS,
    **GRIDPUZZLE_DATA_CONFIGS,
    **PRMBENCH_DATA_CONFIGS,
    **DELTABENCH_DATA_CONFIGS,
    **SYNTHCHAIN_DATA_CONFIGS,
    **NATURALSYNTHCHAIN_DATA_CONFIGS,
    **RECIPE_GRAPH_DATA_CONFIGS,
}

METHOD_CONFIGS = {
    'entail_binary': {
        'method': 'entail',
        'kwargs': {
            'entailment_mode': 'binary',
        },
    },  
    'entail_granular': {
        'method': 'entail',
        'kwargs': {
            'entailment_mode': 'granular',
        },
    },
    'entail_raw_binary': {
        'method': 'entail_raw',
        'kwargs': {
            'entailment_mode': 'binary',
        },
    },
    'entail_raw_granular': {
        'method': 'entail_raw',
        'kwargs': {
            'entailment_mode': 'granular',
        },
    },
    'cert_granular_temp0_exact': {
        'method': 'cert_exact',
        'kwargs': {
            'entailment_mode': 'granular',
        },
    },
    'cert_binary_temp0_exact': {
        'method': 'cert_exact',
        'kwargs': {
            'entailment_mode': 'binary',
        },
    },
    'cert_granular_temp0_nonexact': {
        'method': 'cert_nonexact',
        'kwargs': {
            'epsilon': 0.1,
            'delta': 0.1,
            'entailment_mode': 'granular',
        },
    },
    'cert_binary_temp0_nonexact': {
        'method': 'cert_nonexact',
        'kwargs': {
            'epsilon': 0.1,
            'delta': 0.1,
            'entailment_mode': 'binary',
        },
    },
    'cert_granular_temp0_nonexact_e02_d01': {
        'method': 'cert_nonexact',
        'kwargs': {
            'epsilon': 0.2,
            'delta': 0.1,
            'entailment_mode': 'granular',
        },
    },
    'cert_granular_temp0_nonexact_e03_d01': {
        'method': 'cert_nonexact',
        'kwargs': {
            'epsilon': 0.3,
            'delta': 0.1,
            'entailment_mode': 'granular',
        },
    },
    'cert_granular_temp0_nonexact_e04_d01': {
        'method': 'cert_nonexact',
        'kwargs': {
            'epsilon': 0.4,
            'delta': 0.1,
            'entailment_mode': 'granular',
        },
    },
    'cert_granular_temp0_nonexact_test': {
        'method': 'cert_nonexact',
        'kwargs': {
            'epsilon': 0.3,
            'delta': 0.3,
            'entailment_mode': 'granular',
        },
    },
    'cert_binary_temp0_nonexact_test': {
        'method': 'cert_nonexact',
        'kwargs': {
            'epsilon': 0.3,
            'delta': 0.3,
            'entailment_mode': 'binary',
        },
    },
    'cert_granular_temp0_nonexact_equal': {
        'method': 'cert_nonexact_equal',
        'kwargs': {
            'epsilon': 0.1,
            'delta': 0.1,
            'entailment_mode': 'granular',
        },
    },
    'cert_binary_temp0_nonexact_equal': {
        'method': 'cert_nonexact_equal',
        'kwargs': {
            'epsilon': 0.1,
            'delta': 0.1,
            'entailment_mode': 'binary',
        },
    },
    'cert_binary_temp05_nonexact': {
        'method': 'cert_sample',
        'kwargs': {
            'epsilon': 0.1,
            'delta': 0.1,
            'temperature': 0.5,
            'entailment_mode': 'binary',
            'threshold': 0.6,
        },
    },
    'cert_granular_temp05_nonexact_threshold06_test': { 
        'method': 'cert_sample',
        'kwargs': {
            'epsilon': 0.7,
            'delta': 0.7,
            'temperature': 0.5,
            'entailment_mode': 'granular',
            'threshold': 0.6,
        },
    },
    'cert_granular_temp05_nonexact_threshold06_test2': { 
        'method': 'cert_sample',
        'kwargs': {
            'epsilon': 0.5,
            'delta': 0.3,
            'temperature': 0.5,
            'entailment_mode': 'granular',
            'threshold': 0.6,
        },
    },
    'cert_granular_temp05_nonexact_threshold06': { 
        'method': 'cert_sample',
        'kwargs': {
            'epsilon': 0.1,
            'delta': 0.1,
            'temperature': 0.5,
            'entailment_mode': 'granular',
            'threshold': 0.6,
        },
    },
    'cert_granular_temp05_nonexact_threshold08': { 
        'method': 'cert_sample',
        'kwargs': {
            'epsilon': 0.1,
            'delta': 0.1,
            'temperature': 0.5,
            'entailment_mode': 'granular',
            'threshold': 0.8,
        },
    },
    'cert_granular_temp05_nonexact_threshold10': { 
        'method': 'cert_sample',
        'kwargs': {
            'epsilon': 0.1,
            'delta': 0.1,
            'temperature': 0.5,
            'entailment_mode': 'granular',
            'threshold': 1.0,
        },
    },
    'cert_granular_temp05_nonexact_threshold04': { 
        'method': 'cert_sample',
        'kwargs': {
            'epsilon': 0.1,
            'delta': 0.1,
            'temperature': 0.5,
            'entailment_mode': 'granular',
            'threshold': 0.4,
        },
    },
    'cert_granular_temp05_nonexact_threshold02': { 
        'method': 'cert_sample',
        'kwargs': {
            'epsilon': 0.1,
            'delta': 0.1,
            'temperature': 0.5,
            'entailment_mode': 'granular',
            'threshold': 0.2,
        },
    },
    'llm_judge_whole': {
        'method': 'llm_judge_whole',
        'kwargs': {
            'max_tokens': 5000
        }
    },
    'llm_judge_whole_granular': {
        'method': 'llm_judge_whole',
        'kwargs': {
            'max_tokens': 5000,
            'entailment_mode': 'granular',
        }
    },
    'llm_judge_whole_binary': {
        'method': 'llm_judge_whole',
        'kwargs': {
            'max_tokens': 5000,
            'entailment_mode': 'binary',
        }
    },
    'prm': {
        'method': 'prm',
        'kwargs': {}
    },
    'llm_probs': {
        'method': 'probability',
        'kwargs': {}
    },
    'llm_prompt': {
        'method': 'standard_prompt',
        'kwargs': {}
    },
    'llm_consistency': {
        'method': 'consistency',
        'kwargs': {}
    },
    'llm_chain_of_thought': {
        'method': 'chain_of_thought',
        'kwargs': {}
    },
    'llm_judge_granular': {
        'method': 'llm_judge',
        'kwargs': {
            'entailment_mode': 'granular',
        }
    },
    'llm_judge_binary': {
        'method': 'llm_judge',
        'kwargs': {
            'entailment_mode': 'binary',
        }
    },
    'receval_inter_correct_binary': {
        'method': 'receval_inter_correct',
        'kwargs': {
            'entailment_mode': 'binary',
        }
    },
    'receval_inter_correct_granular': {
        'method': 'receval_inter_correct',
        'kwargs': {
            'entailment_mode': 'granular',
        }
    },
    'receval_intra_correct_binary': {
        'method': 'receval_intra_correct',
        'kwargs': {
            'entailment_mode': 'binary',
        }
    },
    'receval_intra_correct_granular': {
        'method': 'receval_intra_correct',
        'kwargs': {
            'entailment_mode': 'granular',
        }
    },
    'roscoe_li_source_binary': {
        'method': 'roscoe_li_source',
        'kwargs': {
            'entailment_mode': 'binary',
        }
    },
    'roscoe_li_source_granular': {
        'method': 'roscoe_li_source',
        'kwargs': {
            'entailment_mode': 'granular',
        }
    },
    'roscoe_li_self_binary': {
        'method': 'roscoe_li_self',
        'kwargs': {
            'entailment_mode': 'binary',
        }
    },
    'roscoe_li_self_granular': {
        'method': 'roscoe_li_self',
        'kwargs': {
            'entailment_mode': 'granular',
        }
    },
}

MODEL_CONFIGS = {
    'gpt-4o-mini': {
        'model_type': 'openai',
        'model_name': 'gpt-4o-mini',
    },
    'gpt-4o': {
        'model_type': 'openai',
        'model_name': 'gpt-4o',
    },
    'phi-4': {
        'model_type': 'phi',
        'model_name': 'microsoft/phi-4',
    },
    'llama-3.1-8b-instruct': {
        'model_type': 'llama',
        'model_name': 'meta-llama/Llama-3.1-8B-Instruct',
    },
    'qwen2.5-7b-instruct': {
        'model_type': 'qwen',
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
    },
    'flan-t5-xxl': {
        'model_type': 'flan-t5',
        'model_name': 'google/flan-t5-xxl',
    },
    'flan-t5-xl': {
        'model_type': 'flan-t5',
        'model_name': 'google/flan-t5-xl',
    },
    'math-shepherd-mistral-7b-prm': {
        'model_type': 'prm',
        'model_name': 'peiyi9979/math-shepherd-mistral-7b-prm',
    },
    'qwen2.5-math-prm-7b': {
        'model_type': 'qwen_prm',
        'model_name': 'Qwen/Qwen2.5-Math-PRM-7B',
    },
    'qwen2.5-math-prm-7b-vllm': {
        'model_type': 'qwen_prm_vllm',
        'model_name': 'Qwen/Qwen2.5-Math-PRM-7B',
    },
    'reasoneval-7b': {
        'model_type': 'prm',
        'model_name': 'GAIR/ReasonEval-7B',
    },
    'claude-3-5-haiku': {
        'model_type': 'llm_api',
        'model_name': 'claude-3-5-haiku-20241022',
    },
    'claude-3-5-sonnet': {
        'model_type': 'llm_api',
        'model_name': 'claude-3-5-sonnet-20241022',
    },
    'gemini-1.5-flash': {
        'model_type': 'llm_api',
        'model_name': 'gemini-1.5-flash',
    },
    'gemini-1.5-pro': {
        'model_type': 'llm_api',
        'model_name': 'gemini-1.5-pro',
    },
    'qwen3-4b': {
        'model_type': 'vllm',
        'model_name': 'Qwen/Qwen3-4B',
        'use_tqdm': False,
    },
    'qwen3-32b': {
        'model_type': 'vllm',
        'model_name': 'Qwen/Qwen3-32B',
        'use_tqdm': False,
    },
    'qwen3-32b-awq': {
        'model_type': 'vllm',
        'model_name': 'Qwen/Qwen3-32B-AWQ',
        'use_tqdm': False,
    },
    'qwen3-4b-awq': {
        'model_type': 'vllm',
        'model_name': 'Qwen/Qwen3-4B-AWQ',
        'use_tqdm': False,
    },
}

# Parameters that can be varied independently for each experiment
HALLUCINATION_Q_PROBS = [0.01, 0.05, 0.1]
TEMPERATURE_VALUES = [0.0, 0.3, 0.7, 1.0]
DELTA_VALUES = [0.01, 0.05, 0.1]
EPSILON_VALUES = [0.01, 0.05, 0.1]
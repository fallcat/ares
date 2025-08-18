from .eb import EBDataset
from .eb import EBDatasetReplaceGlitch
from .eb import EBDatasetReplaceRawGlitch
from .inli import InliDataset
from .inli import InliDatasetInjectingWorldKnowledge
from .inli import InliDatasetInjectingCotGlitch
from .inli import InliDatasetLLMGeneration
from .gsm8k import GSM8KDataset
from .reveal import RevealDataset
from .gridpuzzle import GridPuzzleDataset
from .prmbench import PRMBenchDataset
from .deltabench import DeltaBenchDataset
from .synthchain import SyntheticChainDataset
from .naturalsynthchain import NaturalSyntheticChainDataset
from .recipe_graph import RecipeGraphDataset
def get_dataset(config, split, root_dir=None):
    """
    Get the dataset from the config
    """
    if config['dataset_type'] == 'eb':
        return EBDataset(config, split, root_dir)
    elif config['dataset_type'] == 'eb_replace_glitch':
        return EBDatasetReplaceGlitch(config, split, root_dir)
    elif config['dataset_type'] == 'eb_replace_raw_glitch':
        return EBDatasetReplaceRawGlitch(config, split, root_dir)
    elif config['dataset_type'] == 'inli':
        return InliDataset(config, split, root_dir)
    elif config['dataset_type'] == 'inli_world_knowledge':
        return InliDatasetInjectingWorldKnowledge(config, split, root_dir)
    elif config['dataset_type'] == 'inli_cot_glitch':
        return InliDatasetInjectingCotGlitch(config, split, root_dir)
    elif config['dataset_type'] == 'inli_llm_generation':
        return InliDatasetLLMGeneration(config, split, root_dir)
    elif config['dataset_type'] == 'gsm8k':
        return GSM8KDataset(config, split)
    elif config['dataset_type'] == 'reveal':
        return RevealDataset(config, split)
    elif config['dataset_type'] == 'gridpuzzle':
        return GridPuzzleDataset(config, split, root_dir)
    elif config['dataset_type'] == 'prmbench':
        return PRMBenchDataset(config, split)
    elif config['dataset_type'] == 'deltabench':
        return DeltaBenchDataset(config, split, root_dir)
    elif config['dataset_type'] == 'synthchain':
        return SyntheticChainDataset(config, split)
    elif config['dataset_type'] == 'naturalsynthchain':
        return NaturalSyntheticChainDataset(config, split)
    elif config['dataset_type'] == 'recipe_graph':
        return RecipeGraphDataset(config, split, root_dir)
    else:
        raise ValueError(f"Dataset type {config['dataset_type']} not supported")

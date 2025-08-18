import random
import string
from .base import BaseDataset
import torch
from torch.utils.data import Dataset
import itertools

DOMAINS = {
        "cooking": [
            "buy ingredients", "prepare vegetables", "boil water", "cook pasta", "drain pasta",
            "make sauce", "mix pasta and sauce", "set the table", "serve food", "eat dinner",
            "feel satisfied", "clean dishes", "wipe table", "store leftovers", "plan next meal",
            "write grocery list", "go shopping", "carry groceries home", "unpack groceries", "rest"
        ],
        "school": [
            "wake up early", "eat breakfast", "go to school", "attend class", "take notes",
            "study notes", "do homework", "ask questions", "prepare for exam", "take exam",
            "pass exam", "get good grades", "get praise", "feel confident", "apply to college",
            "get accepted", "choose major", "attend orientation", "start college", "graduate"
        ],
        "gardening": [
            "choose location", "buy soil", "till soil", "buy seeds", "plant seeds",
            "water seeds", "watch sprouts", "pull weeds", "fertilize plants", "see flowers",
            "attract bees", "grow fruit", "pick fruit", "wash fruit", "prepare jam",
            "can jam", "label jars", "store jam", "share jam", "enjoy jam"
        ],
        "photography": [
            "charge battery", "pack camera", "go outside", "find a scene", "set up tripod",
            "adjust lens", "check lighting", "take photo", "review photo", "adjust settings",
            "take more photos", "capture perfect shot", "go home", "upload photos", "edit photos",
            "sort best shots", "create album", "publish online", "get feedback", "improve skills"
        ],
        "travel": [
            "choose destination", "book flight", "pack bags", "go to airport", "check in",
            "go through security", "board plane", "fly to destination", "arrive at hotel", "check in hotel",
            "unpack bags", "explore city", "visit landmarks", "try local food", "buy souvenirs",
            "take photos", "make memories", "return to hotel", "sleep well", "fly home"
        ]
    }

# Re-run the function to recreate the natural_dataset dictionary
def create_natural_dataset_format(states, seed=42):
    rng = random.Random(seed)
    results = {}

    rule_templates = [
        "Rule: If I {p}, then I can {q}.",
        "Rule: Having {p} enables me to {q}.",
        "Rule: To {q}, I must first {p}.",
        "Rule: When I {p}, I am able to {q}.",
        "Rule: If {p}, then {q}."
    ]

    cot_template = (
        "I {p}. I know that if I {p}, I can {q}. Therefore I can now {q}."
    )

    # for domain, states in domains_dict.items():
    rules = [(p, q) for (p, q) in zip(states[:-1], states[1:])]
    all_raw_claims = [
        rng.choice(rule_templates).format(p=p, q=q)
        for (p, q) in rules
    ] + [f"I {states[0]}."]

    deleted_index = rng.randint(0, len(all_raw_claims) - 1)
    deleted_raw_claim = all_raw_claims[deleted_index]
    remained_raw_claims = all_raw_claims[:deleted_index] + all_raw_claims[deleted_index+1:]
    rng.shuffle(remained_raw_claims)

    derived_claims = [
        cot_template.format(p=p, q=q)
        for (p, q) in rules
    ]

    results = {
        "states": states,
        "rules": rules,
        "deleted_index": deleted_index,
        "all_raw_claims": all_raw_claims,
        "deleted_raw_claim": deleted_raw_claim,
        "remained_raw_claims": remained_raw_claims,
        "derived_claims": derived_claims,
    }

    return results


class NaturalSyntheticChainDatasetRaw(Dataset):
    

    def __init__(self, num_states: int, dataset_len: int = 1000, domain: str = "Cooking", seed: int = 42):
        self.num_states = num_states
        self.all_names = DOMAINS[domain]
        self.dataset_len = dataset_len

    def __len__(self):
        return self.dataset_len

    def shuffle_list_with_local_rng(self, input_list: list, seed: int) -> list:
        local_rng = random.Random(seed)
        list_copy = input_list[:]  # Slicing creates a shallow copy
        local_rng.shuffle(list_copy)
        return list_copy

    def sample_random_integer(self, max_val: int, seed: int) -> int:
        local_rng = random.Random(seed)
        return local_rng.randint(0, max_val)                         
    
    def __getitem__(self, idx):
        return create_natural_dataset_format(self.all_names, seed=idx)


class NaturalSyntheticChainDataset(BaseDataset):
    def __init__(self, config, split):
        super().__init__(config)
        self.split = split
        self.data = self.get_raw_data()

    def get_raw_data(self):
        # task_id is domain: cooking, school, gardening, photography, travel
        return NaturalSyntheticChainDatasetRaw(self.config['num_states'], self.config['dataset_len'], self.config['task_id'], self.config['seed'])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_example = self.data[idx]

        raw_claims = data_example['remained_raw_claims']
        derived_claims = data_example['derived_claims']
        data_entry = self.get_data_entry(raw_claims, derived_claims)
        data_entry['raw_claims'] = raw_claims
        data_entry['derived_claims'] = derived_claims
        data_entry['original_data'] = data_example

        # errors will be starting from the deleted_idx
        if 'error_idxs' in data_example:
            data_entry['error_idxs'] = data_example['error_idxs']
        else:
            data_entry['error_idxs'] = [
                1 if i >= data_example['deleted_index'] else 0
                for i in range(len(derived_claims))
            ]
        return data_entry
import random
import string
from .base import BaseDataset
import torch
from torch.utils.data import Dataset
import itertools

def generate_random_strings(num_samples: int, each_len: int = 4, seed: int = None) -> list[str]:
    """
    Generates a specified number of random strings, each of a given length.
    Each string will:
    - Contain only uppercase English letters (A-Z) and digits (0-9).
    - Begin with an uppercase English letter.
    """
    if each_len <= 0:
        raise ValueError("each_len must be at least 1 to generate a valid string.")

    if num_samples == 0:
        return []

    # Use a local random number generator instead of affecting global state
    local_rng = random.Random(seed) if seed is not None else random.Random()

    # Characters for the first position (must be an uppercase letter)
    valid_start_chars = string.ascii_uppercase
    # Characters for subsequent positions (uppercase letters or digits)
    valid_other_chars = string.ascii_uppercase + string.digits

    generated_strings = []

    for _ in range(num_samples):
        # Generate the first character (must be an uppercase letter)
        first_char = local_rng.choice(valid_start_chars)

        # Generate the remaining characters if each_len > 1
        if each_len == 1:
            candidate_string = first_char
        else:
            # Generate (each_len - 1) characters from uppercase letters or digits
            rest_chars = ''.join(local_rng.choice(valid_other_chars) for _ in range(each_len - 1))
            candidate_string = first_char + rest_chars
        
        generated_strings.append(candidate_string)

    return generated_strings

def generate_random_strings_no_repeat(num_samples: int, each_len: int = 4, seed: int = None) -> list[str]:
    """
    Generates a specified number of random strings, each of a given length.
    Each string will:
    - Contain only uppercase English letters (A-Z) and digits (0-9).
    - Begin with an uppercase English letter.
    """
    if each_len <= 0:
        raise ValueError("each_len must be at least 1 to generate a valid string.")

    if num_samples == 0:
        return []

    # Use a local random number generator instead of affecting global state
    local_rng = random.Random(seed) if seed is not None else random.Random()

    # Characters for the first position (must be an uppercase letter)
    valid_start_chars = string.ascii_uppercase
    # Characters for subsequent positions (uppercase letters or digits)
    valid_other_chars = string.ascii_uppercase + string.digits

    generated_strings = []

    # Calculate the maximum possible unique strings with the given length
    max_possible = 26  # For each_len=1, only 26 uppercase letters
    if each_len > 1:
        max_possible = 26 * (36 ** (each_len - 1))  # 26 first chars, 36 chars for remaining positions
    
    # If we're requesting all or most possible strings and each_len is small enough,
    # generate all possibilities and sample from them
    if each_len <= 3 and num_samples <= max_possible:
        # Generate all possible strings systematically
        all_possible_strings = []
        
        # First character must be uppercase letter
        for first in valid_start_chars:
            if each_len == 1:
                all_possible_strings.append(first)
            else:
                # For remaining positions, use all valid characters
                for combo in itertools.product(valid_other_chars, repeat=each_len-1):
                    all_possible_strings.append(first + ''.join(combo))
        
        # Sample without replacement from all possibilities
        generated_strings = local_rng.sample(all_possible_strings, num_samples)
    else:
        # For longer strings or when requesting a small subset, use random generation with duplicate checking
        unique_strings = set()
        
        # Try to generate the requested number of unique strings
        attempts = 0
        max_attempts = num_samples * 10  # Limit attempts to prevent infinite loops
        
        while len(unique_strings) < num_samples and attempts < max_attempts:
            # Generate the first character (must be an uppercase letter)
            first_char = local_rng.choice(valid_start_chars)

            # Generate the remaining characters if each_len > 1
            if each_len == 1:
                candidate_string = first_char
            else:
                # Generate (each_len - 1) characters from uppercase letters or digits
                rest_chars = ''.join(local_rng.choice(valid_other_chars) for _ in range(each_len - 1))
                candidate_string = first_char + rest_chars
            
            # Only add the string if it's not already in our set
            if candidate_string not in unique_strings:
                unique_strings.add(candidate_string)
                generated_strings.append(candidate_string)
            
            attempts += 1
        
        # If we couldn't generate enough unique strings, warn the user
        if len(generated_strings) < num_samples:
            print(f"Warning: Could only generate {len(generated_strings)} unique strings instead of the requested {num_samples}.")

    return generated_strings


class SyntheticChainDatasetRaw(Dataset):
    def __init__(self, num_states: int, dataset_len: int = 1000, each_len: int = 4, seed: int = 42):
        self.num_states = num_states
        self.all_names = generate_random_strings(num_states, each_len=each_len, seed=seed)
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
        states = self.shuffle_list_with_local_rng(self.all_names, idx)
        rules = [(p, q) for (p, q) in zip(states[:-1], states[1:])]
        all_raw_claims = [
            f"{p} -> {q}"
            for (p, q) in rules
        ] + [f"I have {states[0]}"]

        deleted_index = self.sample_random_integer(len(all_raw_claims)-1, seed=idx)
        deleted_raw_claim = all_raw_claims[deleted_index]
        remained_raw_claims = all_raw_claims[:deleted_index] + all_raw_claims[deleted_index+1:]
        remained_raw_claims = self.shuffle_list_with_local_rng(remained_raw_claims, idx+1)
        
        derived_claims : list[str] = [
            f"I use rule ({p} -> {q}) to derive {q}"
            for (p, q) in zip(states[:-1], states[1:])
        ]
        
        return {
            "states": states,
            "rules": rules,
            "deleted_index": deleted_index,
            "all_raw_claims": all_raw_claims,
            "deleted_raw_claim": deleted_raw_claim,
            "remained_raw_claims": remained_raw_claims,
            "derived_claims": derived_claims,
        }    

class SyntheticChainDatasetRaw2(Dataset):
    def __init__(self, num_states: int, dataset_len: int = 1000, each_len: int = 4, seed: int = 42):
        self.num_states = num_states
        self.all_names = generate_random_strings(num_states, each_len=each_len, seed=seed)
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
        states = self.shuffle_list_with_local_rng(self.all_names, idx)
        rules = [(p, q) for (p, q) in zip(states[:-1], states[1:])]
        all_raw_claims = [
            f"Rule:{p} -> {q}"
            for (p, q) in rules
        ] + [f"I have {states[0]}"]

        deleted_index = self.sample_random_integer(len(all_raw_claims)-1, seed=idx)
        deleted_raw_claim = all_raw_claims[deleted_index]
        remained_raw_claims = all_raw_claims[:deleted_index] + all_raw_claims[deleted_index+1:]
        remained_raw_claims = self.shuffle_list_with_local_rng(remained_raw_claims, idx+1)
        
        derived_claims : list[str] = [
            f"I have {p}, I use rule ({p} -> {q}) to derive {q}"
            for (p, q) in zip(states[:-1], states[1:])
        ]
        
        return {
            "states": states,
            "rules": rules,
            "deleted_index": deleted_index,
            "all_raw_claims": all_raw_claims,
            "deleted_raw_claim": deleted_raw_claim,
            "remained_raw_claims": remained_raw_claims,
            "derived_claims": derived_claims,
        }        

class SyntheticChainDatasetRaw3(Dataset):
    def __init__(self, num_states: int, dataset_len: int = 1000, each_len: int = 4, seed: int = 42):
        self.num_states = num_states
        self.all_names = generate_random_strings(num_states, each_len=each_len, seed=seed)
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
        states = self.shuffle_list_with_local_rng(self.all_names, idx)
        rules = [(p, q) for (p, q) in zip(states[:-1], states[1:])]
        all_raw_claims = [
            f"Rule: {p} -> {q} (meaning that if I have {p}, I can derive {q})"
            for (p, q) in rules
        ] + [f"I have {states[0]}"]

        deleted_index = self.sample_random_integer(len(all_raw_claims)-1, seed=idx)
        deleted_raw_claim = all_raw_claims[deleted_index]
        remained_raw_claims = all_raw_claims[:deleted_index] + all_raw_claims[deleted_index+1:]
        remained_raw_claims = self.shuffle_list_with_local_rng(remained_raw_claims, idx+1)
        
        derived_claims : list[str] = [
            f"I have {p}, I use rule ({p} -> {q}) to derive {q}, now I have {q}"
            for (p, q) in zip(states[:-1], states[1:])
        ]
        
        return {
            "states": states,
            "rules": rules,
            "deleted_index": deleted_index,
            "all_raw_claims": all_raw_claims,
            "deleted_raw_claim": deleted_raw_claim,
            "remained_raw_claims": remained_raw_claims,
            "derived_claims": derived_claims,
        }  

class SyntheticChainDatasetRaw3NoRepeat(Dataset):
    def __init__(self, num_states: int, dataset_len: int = 1000, each_len: int = 4, seed: int = 42):
        self.num_states = num_states
        self.all_names = generate_random_strings_no_repeat(num_states, each_len=each_len, seed=seed)
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
        states = self.shuffle_list_with_local_rng(self.all_names, idx)
        rules = [(p, q) for (p, q) in zip(states[:-1], states[1:])]
        all_raw_claims = [
            f"Rule: {p} -> {q} (meaning that if I have {p}, I can derive {q})"
            for (p, q) in rules
        ] + [f"I have {states[0]}"]

        deleted_index = self.sample_random_integer(len(all_raw_claims)-1, seed=idx)
        deleted_raw_claim = all_raw_claims[deleted_index]
        remained_raw_claims = all_raw_claims[:deleted_index] + all_raw_claims[deleted_index+1:]
        remained_raw_claims = self.shuffle_list_with_local_rng(remained_raw_claims, idx+1)
        
        derived_claims : list[str] = [
            f"I have {p}, I use rule ({p} -> {q}) to derive {q}, now I have {q}"
            for (p, q) in zip(states[:-1], states[1:])
        ]
        
        return {
            "states": states,
            "rules": rules,
            "deleted_index": deleted_index,
            "all_raw_claims": all_raw_claims,
            "deleted_raw_claim": deleted_raw_claim,
            "remained_raw_claims": remained_raw_claims,
            "derived_claims": derived_claims,
        }  

class SyntheticChainDatasetRaw4(Dataset):
    def __init__(self, num_states: int, dataset_len: int = 1000, each_len: int = 4, seed: int = 42):
        self.num_states = num_states
        self.all_names = generate_random_strings(num_states, each_len=each_len, seed=seed)
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
        states = self.shuffle_list_with_local_rng(self.all_names, idx)
        rules = [(p, q) for (p, q) in zip(states[:-1], states[1:])]
        all_raw_claims = [
            f"Existing Rule: {p} -> {q} (meaning that if I have {p}, I can derive {q})"
            for (p, q) in rules
        ] + [f"I already have {states[0]}"]

        deleted_index = self.sample_random_integer(len(all_raw_claims)-1, seed=idx)
        deleted_raw_claim = all_raw_claims[deleted_index]
        remained_raw_claims = all_raw_claims[:deleted_index] + all_raw_claims[deleted_index+1:]
        remained_raw_claims = self.shuffle_list_with_local_rng(remained_raw_claims, idx+1)
        
        derived_claims : list[str] = [
            f"I have {p}, I use an existing rule ({p} -> {q}) to derive {q}, now I have {q}"
            for (p, q) in zip(states[:-1], states[1:])
        ]
        
        return {
            "states": states,
            "rules": rules,
            "deleted_index": deleted_index,
            "all_raw_claims": all_raw_claims,
            "deleted_raw_claim": deleted_raw_claim,
            "remained_raw_claims": remained_raw_claims,
            "derived_claims": derived_claims,
        }     

class SyntheticChainDatasetRaw5(Dataset):
    def __init__(self, num_states: int, dataset_len: int = 1000, each_len: int = 4, seed: int = 42):
        self.num_states = num_states
        self.all_names = generate_random_strings(num_states, each_len=each_len, seed=seed)
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
        states = self.shuffle_list_with_local_rng(self.all_names, idx)
        rules = [(p, q) for (p, q) in zip(states[:-1], states[1:])]
        all_raw_claims = [
            f"Existing Rule: {p} -> {q} (meaning that if I have {p}, I can derive {q})"
            for (p, q) in rules
        ] + [f"I already have {states[0]}"]

        deleted_index = self.sample_random_integer(len(all_raw_claims)-1, seed=idx)
        deleted_raw_claim = all_raw_claims[deleted_index]
        remained_raw_claims = all_raw_claims[:deleted_index] + all_raw_claims[deleted_index+1:]
        remained_raw_claims = self.shuffle_list_with_local_rng(remained_raw_claims, idx+1)
        
        derived_claims : list[str] = [
            f"I have {p}, and there is an existing rule ({p} -> {q}), so I can derive {q}, now I have {q}"
            for (p, q) in zip(states[:-1], states[1:])
        ]
        
        return {
            "states": states,
            "rules": rules,
            "deleted_index": deleted_index,
            "all_raw_claims": all_raw_claims,
            "deleted_raw_claim": deleted_raw_claim,
            "remained_raw_claims": remained_raw_claims,
            "derived_claims": derived_claims,
        }     

class SyntheticChainDatasetMultiSourcesSingleSink(Dataset):
    """
    We want to test the effect of irrelevant rules that can also point to the sink.
    """
    def __init__(self, num_sources: int = 3, depth: int = 3, dataset_len: int = 1000, each_len: int = 4, seed: int = 42):
        self.num_sources = num_sources
        self.depth = depth
        self.num_states = num_sources * (depth - 1) + 1 # all states go to one sink
        self.all_names = generate_random_strings_no_repeat(self.num_states, each_len=each_len, seed=seed)
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
        states = self.shuffle_list_with_local_rng(self.all_names, idx)
        rules = [] # there are num_sources * (depth - 1) rules
        # the last state is the sink for all sources
        # A -> B -> C -> D -> E -> ... -> T
        # A'-> B'-> C'-> D'-> E'-> ... -> T
        # A''-> B''-> C''-> D''-> E''-> ... -> T
        # ...   
        sources = []
        for source in range(self.num_sources):
            sources.append(states[source * (self.depth - 1)])
            for depth in range(self.depth - 2): # need to stop before the last state because for pairs we need rules that have num_pairs - 1 rules
                rules.append((states[source * (self.depth - 1) + depth], states[source * (self.depth - 1) + depth + 1]))
            # from the last state in this chain, go to the sink
            rules.append((states[source * (self.depth - 1) + self.depth - 2], states[-1]))
        sinks = [states[-1]]
        # rules = [(p, q) for (p, q) in zip(states[:-1], states[1:])]
        all_raw_claims = [
            f"Rule: {p} -> {q} (meaning that if I have {p}, I can derive {q})"
            for (p, q) in rules
        ] + [f"I have {states[0]}"] # only ones in the first chain are correct because only the first source is given, others are incorrect. after deletion are also incorrect.

        deleted_index = self.sample_random_integer(len(all_raw_claims)-1, seed=idx) # if this is in the first chain, then it will affect later ones.
        deleted_raw_claim = all_raw_claims[deleted_index]
        remained_raw_claims = all_raw_claims[:deleted_index] + all_raw_claims[deleted_index+1:]
        remained_raw_claims = self.shuffle_list_with_local_rng(remained_raw_claims, idx+1)
        
        derived_claims : list[str] = [
            f"I have {p}, I use rule ({p} -> {q}) to derive {q}, now I have {q}"
            for (p, q) in zip(states[:self.depth - 1], states[1:self.depth - 1] + [states[-1]]) # the first chain
        ] # A -> B, B -> C, (C -> D), D -> E, E -> T (deleted_idx=2)
        error_idxs = [
            1 if i >= deleted_index else 0
            for i in range(len(derived_claims))
        ]
        
        return {
            "states": states,
            "rules": rules,
            "sources": sources,
            "sinks": sinks,
            "deleted_index": deleted_index,
            "all_raw_claims": all_raw_claims,
            "deleted_raw_claim": deleted_raw_claim,
            "remained_raw_claims": remained_raw_claims,
            "derived_claims": derived_claims,
            "error_idxs": error_idxs,
        }  

class SyntheticChainDatasetInsertion(Dataset):
    """
    We want to test the effect of insertion of non-existing rules.
    This is only inserting one step errors each time.
    """
    def __init__(self, num_valid_states: int, num_insertions: int, dataset_len: int = 1000, each_len: int = 4, seed: int = 42):
        self.num_valid_states = num_valid_states
        self.num_insertions = num_insertions
        self.num_states = num_valid_states + num_insertions
        self.all_names = generate_random_strings_no_repeat(self.num_states, each_len=each_len, seed=seed)
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
        states = self.shuffle_list_with_local_rng(self.all_names, idx)
        rules = [(p, q) for (p, q) in zip(states[:self.num_valid_states - 1], states[1:self.num_valid_states])] # only include rules in valid states
        all_raw_claims = [
            f"Rule: {p} -> {q} (meaning that if I have {p}, I can derive {q})"
            for (p, q) in rules
        ] + [f"I have {states[0]}"]

        insertion_indices = [self.sample_random_integer(len(all_raw_claims)-2, seed=(idx+ii)) for ii in range(self.num_insertions)]

        derived_claims = []
        error_idxs = []

        for i in range(self.num_valid_states - 1):
            derived_claims.append(f"I have {states[i]}, I use rule ({states[i]} -> {states[i+1]}) to derive {states[i+1]}, now I have {states[i+1]}")
            error_idxs.append(0)
            if i in insertion_indices:
                # get all indices of i in insertion_indices
                i_indices = [j for j, x in enumerate(insertion_indices) if x == i]
                for j in i_indices: # there can be multiple insertions for the same state
                    p = states[i]
                    try:
                        q = states[self.num_valid_states + j] # the corresponding insertion state will be an invalid state
                        derived_claims.append(f"I have {p}, I use rule ({p} -> {q}) to derive {q}, now I have {q}")
                        error_idxs.append(1)
                    except:
                        import pdb; pdb.set_trace()
                        print(f"Error: {p} -> {q}")
                        print(f"insertion_indices: {insertion_indices}")
                        print(f"i: {i}")
                        print(f"j: {j}")
                        print(f"states: {states}")
                        

        remained_raw_claims = self.shuffle_list_with_local_rng(all_raw_claims, idx+1)
        
        return {
            "states": states,
            "rules": rules,
            "all_raw_claims": all_raw_claims,
            "remained_raw_claims": remained_raw_claims,
            "derived_claims": derived_claims,
            "error_idxs": error_idxs,
            "insertion_indices": insertion_indices,
        }  

class SyntheticChainDataset(BaseDataset):
    def __init__(self, config, split):
        super().__init__(config)
        self.split = split
        self.data = self.get_raw_data()

    def get_raw_data(self):
        if self.config['task_id'] == 'vanilla':
            return SyntheticChainDatasetRaw(self.config['num_states'], self.config['dataset_len'], self.config['each_len'], self.config['seed'])
        elif self.config['task_id'] == 'vanilla2':
            return SyntheticChainDatasetRaw2(self.config['num_states'], self.config['dataset_len'], self.config['each_len'], self.config['seed'])
        elif self.config['task_id'] == 'vanilla3':
            return SyntheticChainDatasetRaw3(self.config['num_states'], self.config['dataset_len'], self.config['each_len'], self.config['seed'])
        elif self.config['task_id'] == 'vanilla4':
            return SyntheticChainDatasetRaw4(self.config['num_states'], self.config['dataset_len'], self.config['each_len'], self.config['seed'])
        elif self.config['task_id'] == 'vanilla5':
            return SyntheticChainDatasetRaw5(self.config['num_states'], self.config['dataset_len'], self.config['each_len'], self.config['seed'])
        elif self.config['task_id'] == 'vanilla3_no_repeat':
            return SyntheticChainDatasetRaw3NoRepeat(self.config['num_states'], self.config['dataset_len'], self.config['each_len'], self.config['seed'])
        elif self.config['task_id'] == 'vanilla_multi_sources_single_sink':
            return SyntheticChainDatasetMultiSourcesSingleSink(self.config['num_sources'], self.config['depth'], self.config['dataset_len'], self.config['each_len'], self.config['seed'])
        elif self.config['task_id'] == 'insertion':
            return SyntheticChainDatasetInsertion(self.config['num_valid_states'], self.config['num_insertions'], self.config['dataset_len'], self.config['each_len'], self.config['seed'])
        else:
            raise ValueError(f"Invalid task_id: {self.config['task_id']}")

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
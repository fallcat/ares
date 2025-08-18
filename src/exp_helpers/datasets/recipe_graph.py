import random
import string
from .base import BaseDataset
import torch
from torch.utils.data import Dataset
import itertools
from collections import defaultdict
import math
import os
import json
from collections import deque

class RecipeGraphDatasetRaw(Dataset):
    def __init__(self, data_dir: str, dataset_len: int = 1000, seed: int = 42):
        self.data_dir = data_dir
        self.dataset_len = dataset_len
        self.seed = seed

        # get data
        self.data = []
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith('.json'):
                with open(os.path.join(data_dir, filename), 'r') as f:
                    data = json.load(f)
                    self.data.append(data)

        # repeat the data to be dataset_len
        self.data = self.data * (math.ceil(dataset_len / len(self.data)))
        self.data = self.data[:dataset_len]

        # shuffle data
        self.data = self.shuffle_list_with_local_rng(self.data, self.seed)

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
        data_example = self.data[idx]

        steps = {int(k): v for k, v in data_example['steps'].items()}
        edges = [tuple([p, q]) for (p, q) in data_example['edges']]

        topo_order = data_example['topo_order']
        ingredients = data_example['ingredients'] # this is not in topo_order, but original order
        all_ingredients = data_example['all_ingredients']
        ingredient_mapping = data_example['ingredient_mapping'] 
        all_mapped_ingredients = data_example['all_mapped_ingredients']
        all_mapped_ingredients_unique = data_example['all_mapped_ingredients_unique']
        step_mapped_ingredients = data_example['step_mapped_ingredients'] # this is in original order
        step_mapped_ingredients_dict = {key: step_mapped_ingredients[i] for i, key in enumerate(sorted(steps.keys()))}
        ingredients_dict = {key: ingredients[i] for i, key in enumerate(sorted(steps.keys()))}


        src2tgt = defaultdict(list)
        tgt2src = defaultdict(list)

        for edge in edges:
            src2tgt[edge[0]].append(edge[1])
            tgt2src[edge[1]].append(edge[0])
            
        templates = ['Only after the necessary preceding steps ({}), And if we have all the ingredients, we can then {}.']
        template = templates[0]

        rules = [template.format(', and '.join([steps[nid] for nid in v]), steps[k]) for k, v in tgt2src.items()]

        ingredient_states = [f'We have {item}.' for item in all_mapped_ingredients_unique]

        deleted_ingredient_idx = self.sample_random_integer(len(ingredient_states)-1, self.seed) #random.sample(range(len(ingredient_states)), k=1)[0] # replace
        deleted_ingredient = ingredient_states[deleted_ingredient_idx]
        new_ingredient_states = ingredient_states[:deleted_ingredient_idx] + ingredient_states[deleted_ingredient_idx + 1:]
        remained_raw_claims = rules + new_ingredient_states
        remained_raw_claims = self.shuffle_list_with_local_rng(remained_raw_claims, idx+1) + ['We now START.']

        initial_error_step_ids = []
        for step_id in steps.keys():
            if ingredients_dict[step_id] is not None:
                if all_mapped_ingredients_unique[deleted_ingredient_idx] in step_mapped_ingredients_dict[step_id]:
                    initial_error_step_ids.append(step_id) # this is the index in the topo_order


        all_error_step_ids = []

        dq = deque(initial_error_step_ids)

        while len(dq) > 0:
            step_id = dq.popleft()
            all_error_step_ids.append(step_id)
            downstream_error_step_ids = src2tgt[step_id] # item is in topo_order, but src2tgt is in step_ids
            if downstream_error_step_ids is None:
                continue
            for dstep in downstream_error_step_ids:
                if dstep not in all_error_step_ids:
                    dq.append(dstep)

        initial_error_step_idxs = [topo_order.index(i) for i in initial_error_step_ids]
        all_error_step_idxs = [topo_order.index(i) for i in all_error_step_ids]

        derived_claims = [f'We can now {steps[nid]}.' for nid in topo_order[1:]]
        step_error_labels = [1 if i + 1 in all_error_step_idxs else 0 for i in range(len(derived_claims))] # start is already put in the initial_error_step_idxs

        return {
            "remained_raw_claims": remained_raw_claims,
            "derived_claims": derived_claims,
            "step_error_labels": step_error_labels,
            'initial_error_step_idxs': initial_error_step_idxs,
            'initial_error_step_ids': initial_error_step_ids,
            'all_error_step_idxs': all_error_step_idxs,
            'all_error_step_ids': all_error_step_ids,
            'deleted_ingredient_idx': deleted_ingredient_idx,
            'deleted_ingredient': deleted_ingredient,
            'ingredient_states': ingredient_states,
            'new_ingredient_states': new_ingredient_states,
            'step_mapped_ingredients_dict': step_mapped_ingredients_dict,
            'all_mapped_ingredients_unique': all_mapped_ingredients_unique,
            'all_mapped_ingredients': all_mapped_ingredients,
            'ingredient_mapping': ingredient_mapping,
            'all_ingredients': all_ingredients,
            'steps': steps,
            'edges': edges,
            'topo_order': topo_order,
            'ingredients_dict': ingredients_dict,
            'src2tgt': src2tgt,
            'tgt2src': tgt2src,
        }  

class RecipeGraphDatasetRaw2(Dataset):
    def __init__(self, data_dir: str, dataset_len: int = 1000, seed: int = 42):
        self.data_dir = data_dir
        self.dataset_len = dataset_len
        self.seed = seed

        # get data
        self.data = []
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith('.json'):
                with open(os.path.join(data_dir, filename), 'r') as f:
                    data = json.load(f)
                    self.data.append(data)

        # repeat the data to be dataset_len
        self.data = self.data * (math.ceil(dataset_len / len(self.data)))
        self.data = self.data[:dataset_len]

        # shuffle data
        self.data = self.shuffle_list_with_local_rng(self.data, self.seed)

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
        data_example = self.data[idx]

        steps = {int(k): v for k, v in data_example['steps'].items()}
        edges = [tuple([p, q]) for (p, q) in data_example['edges']]

        topo_order = data_example['topo_order']
        ingredients = data_example['ingredients'] # this is not in topo_order, but original order
        all_ingredients = data_example['all_ingredients']
        ingredient_mapping = data_example['ingredient_mapping'] 
        all_mapped_ingredients = data_example['all_mapped_ingredients']
        all_mapped_ingredients_unique = data_example['all_mapped_ingredients_unique']
        step_mapped_ingredients = data_example['step_mapped_ingredients'] # this is in original order
        step_mapped_ingredients_dict = {key: step_mapped_ingredients[i] for i, key in enumerate(sorted(steps.keys()))}
        ingredients_dict = {key: ingredients[i] for i, key in enumerate(sorted(steps.keys()))}


        src2tgt = defaultdict(list)
        tgt2src = defaultdict(list)

        for edge in edges:
            src2tgt[edge[0]].append(edge[1])
            tgt2src[edge[1]].append(edge[0])
            
        templates = ['Only after the necessary preceding steps ({}), And if we have all the ingredients, we can then {}.']
        template = templates[0]

        rules = [template.format(', and '.join([steps[nid] for nid in v]), steps[k]) for k, v in tgt2src.items()]

        ingredient_states = [f'We have {item}.' for item in all_mapped_ingredients_unique]

        deleted_ingredient_idx = self.sample_random_integer(len(ingredient_states)-1, self.seed) #random.sample(range(len(ingredient_states)), k=1)[0] # replace
        deleted_ingredient = ingredient_states[deleted_ingredient_idx]
        new_ingredient_states = ingredient_states[:deleted_ingredient_idx] + ingredient_states[deleted_ingredient_idx + 1:]
        remained_raw_claims = rules + new_ingredient_states
        remained_raw_claims = self.shuffle_list_with_local_rng(remained_raw_claims, idx+1) + ['We now START.']

        initial_error_step_ids = []
        for step_id in steps.keys():
            if ingredients_dict[step_id] is not None:
                if all_mapped_ingredients_unique[deleted_ingredient_idx] in step_mapped_ingredients_dict[step_id]:
                    initial_error_step_ids.append(step_id) # this is the index in the topo_order


        all_error_step_ids = []

        dq = deque(initial_error_step_ids)

        while len(dq) > 0:
            step_id = dq.popleft()
            all_error_step_ids.append(step_id)
            downstream_error_step_ids = src2tgt[step_id] # item is in topo_order, but src2tgt is in step_ids
            if downstream_error_step_ids is None:
                continue
            for dstep in downstream_error_step_ids:
                if dstep not in all_error_step_ids:
                    dq.append(dstep)

        initial_error_step_idxs = [topo_order.index(i) for i in initial_error_step_ids]
        all_error_step_idxs = [topo_order.index(i) for i in all_error_step_ids]

        derived_claims = [f'We can now do the step {steps[nid]}, because we have all necessary ingredients, and we have already done the preceding steps. And now after we do this step, we have completed the step {steps[nid]}.' for nid in topo_order[1:]]
        step_error_labels = [1 if i + 1 in all_error_step_idxs else 0 for i in range(len(derived_claims))] # start is already put in the initial_error_step_idxs

        return {
            "remained_raw_claims": remained_raw_claims,
            "derived_claims": derived_claims,
            "step_error_labels": step_error_labels,
            'initial_error_step_idxs': initial_error_step_idxs,
            'initial_error_step_ids': initial_error_step_ids,
            'all_error_step_idxs': all_error_step_idxs,
            'all_error_step_ids': all_error_step_ids,
            'deleted_ingredient_idx': deleted_ingredient_idx,
            'deleted_ingredient': deleted_ingredient,
            'ingredient_states': ingredient_states,
            'new_ingredient_states': new_ingredient_states,
            'step_mapped_ingredients_dict': step_mapped_ingredients_dict,
            'all_mapped_ingredients_unique': all_mapped_ingredients_unique,
            'all_mapped_ingredients': all_mapped_ingredients,
            'ingredient_mapping': ingredient_mapping,
            'all_ingredients': all_ingredients,
            'steps': steps,
            'edges': edges,
            'topo_order': topo_order,
            'ingredients_dict': ingredients_dict,
            'src2tgt': src2tgt,
            'tgt2src': tgt2src,
        }  

class RecipeGraphDatasetRaw3(Dataset):
    def __init__(self, data_dir: str, dataset_len: int = 1000, seed: int = 42):
        self.data_dir = data_dir
        self.dataset_len = dataset_len
        self.seed = seed

        # get data
        self.data = []
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith('.json'):
                with open(os.path.join(data_dir, filename), 'r') as f:
                    data = json.load(f)
                    self.data.append(data)

        # repeat the data to be dataset_len
        self.data = self.data * (math.ceil(dataset_len / len(self.data)))
        self.data = self.data[:dataset_len]

        # shuffle data
        self.data = self.shuffle_list_with_local_rng(self.data, self.seed)

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
        data_example = self.data[idx]

        steps = {int(k): v for k, v in data_example['steps'].items()}
        edges = [tuple([p, q]) for (p, q) in data_example['edges']]

        topo_order = data_example['topo_order']
        ingredients = data_example['ingredients'] # this is not in topo_order, but original order
        all_ingredients = data_example['all_ingredients']
        ingredient_mapping = data_example['ingredient_mapping'] 
        all_mapped_ingredients = data_example['all_mapped_ingredients']
        all_mapped_ingredients_unique = data_example['all_mapped_ingredients_unique']
        if 'None' in all_mapped_ingredients_unique:
            all_mapped_ingredients_unique.remove('None')
        step_mapped_ingredients = data_example['step_mapped_ingredients'] # this is in original order
        step_mapped_ingredients_dict = {key: step_mapped_ingredients[i] if step_mapped_ingredients[i] is not None and step_mapped_ingredients[i][0] != 'None' else None for i, key in enumerate(sorted(steps.keys()))}
        ingredients_dict = {key: ingredients[i] if ingredients[i] is not None and ingredients[i][0] != 'None' else None for i, key in enumerate(sorted(steps.keys()))}


        src2tgt = defaultdict(list)
        tgt2src = defaultdict(list)

        for edge in edges:
            src2tgt[edge[0]].append(edge[1])
            tgt2src[edge[1]].append(edge[0])
            
        templates = ['Only after the necessary preceding steps ({}), And if we have all the ingredients, we can then {}.']
        template = templates[0]

        rules = [template.format(', and '.join([steps[nid] for nid in v]), steps[k]) for k, v in tgt2src.items()]

        ingredient_states = [f'We have {item}.' for item in all_mapped_ingredients_unique]

        deleted_ingredient_idx = self.sample_random_integer(len(ingredient_states)-1, self.seed) #random.sample(range(len(ingredient_states)), k=1)[0] # replace
        deleted_ingredient = ingredient_states[deleted_ingredient_idx]
        new_ingredient_states = ingredient_states[:deleted_ingredient_idx] + ingredient_states[deleted_ingredient_idx + 1:]
        remained_raw_claims = rules + new_ingredient_states
        remained_raw_claims = self.shuffle_list_with_local_rng(remained_raw_claims, idx+1) + ['We now START.']

        initial_error_step_ids = []
        for step_id in steps.keys():
            if ingredients_dict[step_id] is not None:
                if all_mapped_ingredients_unique[deleted_ingredient_idx] in step_mapped_ingredients_dict[step_id]:
                    initial_error_step_ids.append(step_id) # this is the index in the topo_order


        all_error_step_ids = []

        dq = deque(initial_error_step_ids)

        while len(dq) > 0:
            step_id = dq.popleft()
            all_error_step_ids.append(step_id)
            downstream_error_step_ids = src2tgt[step_id] # item is in topo_order, but src2tgt is in step_ids
            if downstream_error_step_ids is None:
                continue
            for dstep in downstream_error_step_ids:
                if dstep not in all_error_step_ids:
                    dq.append(dstep)

        initial_error_step_idxs = [topo_order.index(i) for i in initial_error_step_ids]
        all_error_step_idxs = [topo_order.index(i) for i in all_error_step_ids]

        derived_claims = []
        for nid in topo_order[1:]:
            # if there are previous steps, we need to check if we have completed all of them
            previous_steps_str = ''
            if nid in tgt2src and len(tgt2src[nid]) > 0:
                previous_steps_str = ', and '.join([steps[p] for p in tgt2src[nid]])
            previous_steps_str = f'Because we have completed all previous steps ({previous_steps_str}),'

            if ingredients_dict[nid] is not None:
                necessary_ingredients_str = f'and have all necessary ingredients ({", and ".join(ingredients_dict[nid])}),'
            else:
                necessary_ingredients_str = ''

            derived_claims.append(f'{previous_steps_str} {necessary_ingredients_str} we can now do the step {steps[nid]}. And now we have completed this step {steps[nid]}.')
            
        step_error_labels = [1 if i + 1 in all_error_step_idxs else 0 for i in range(len(derived_claims))] # start is already put in the initial_error_step_idxs

        return {
            "remained_raw_claims": remained_raw_claims,
            "derived_claims": derived_claims,
            "step_error_labels": step_error_labels,
            'initial_error_step_idxs': initial_error_step_idxs,
            'initial_error_step_ids': initial_error_step_ids,
            'all_error_step_idxs': all_error_step_idxs,
            'all_error_step_ids': all_error_step_ids,
            'deleted_ingredient_idx': deleted_ingredient_idx,
            'deleted_ingredient': deleted_ingredient,
            'ingredient_states': ingredient_states,
            'new_ingredient_states': new_ingredient_states,
            'step_mapped_ingredients_dict': step_mapped_ingredients_dict,
            'all_mapped_ingredients_unique': all_mapped_ingredients_unique,
            'all_mapped_ingredients': all_mapped_ingredients,
            'ingredient_mapping': ingredient_mapping,
            'all_ingredients': all_ingredients,
            'steps': steps,
            'edges': edges,
            'topo_order': topo_order,
            'ingredients_dict': ingredients_dict,
            'src2tgt': src2tgt,
            'tgt2src': tgt2src,
        }  

class RecipeGraphDataset(BaseDataset):
    def __init__(self, config, split, root_dir: str):
        super().__init__(config)
        self.split = split
        self.root_dir = root_dir
        self.data = self.get_raw_data()

    def get_raw_data(self):
        if self.config['task_id'] == 'recipe_graph':
            return RecipeGraphDatasetRaw(os.path.join(self.root_dir, self.config['data_dir']), self.config['dataset_len'], self.config['seed'])
        elif self.config['task_id'] == 'recipe_graph2':
            return RecipeGraphDatasetRaw2(os.path.join(self.root_dir, self.config['data_dir']), self.config['dataset_len'], self.config['seed'])
        elif self.config['task_id'] == 'recipe_graph3':
            return RecipeGraphDatasetRaw3(os.path.join(self.root_dir, self.config['data_dir']), self.config['dataset_len'], self.config['seed'])
        else:
            raise ValueError(f'Dataset {self.config["task_id"]} not found')

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
        data_entry['step_error_labels'] = data_example['step_error_labels'] # use this label, 1 is error, 0 is correct
        data_entry['initial_error_step_idxs'] = data_example['initial_error_step_idxs']
        data_entry['all_error_step_idxs'] = data_example['all_error_step_idxs']
        data_entry['initial_error_step_ids'] = data_example['initial_error_step_ids']
        data_entry['all_error_step_ids'] = data_example['all_error_step_ids']
        data_entry['deleted_ingredient_idx'] = data_example['deleted_ingredient_idx']
        data_entry['deleted_ingredient'] = data_example['deleted_ingredient']
        return data_entry
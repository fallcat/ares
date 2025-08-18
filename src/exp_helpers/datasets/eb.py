from .base import BaseDataset
import os
import json
import copy
from collections import defaultdict
import random
import re
class EBDataset(BaseDataset):
    def __init__(self, config, split, root_dir):
        """
        Initialize the dataset
        """
        super().__init__(config)
        self.split = split
        self.root_dir = root_dir
        self.data = self.get_raw_data()

    def get_raw_data(self):
        """
        Get the raw data from the dataset
        """
        data = []
        split = self.split
        if split == 'val':
            split = 'dev'
        data_path = os.path.join(self.root_dir, self.config['data_dir'], f'{split}.jsonl')
        # Open the .jsonl file and read line by line
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Parse each line as a JSON object
                data.append(json.loads(line.strip()))

        return data

    def __len__(self):
        """
        Get the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a processed example from the dataset
        """
        data_example = self.data[idx]
        
        sents = data_example['meta']['triples']
        if 'intermediate_conclusions' in data_example['meta']:
            all_hyps = data_example['meta']['intermediate_conclusions']
        else:
            all_hyps = {'hypothesis': data_example['meta']['hypothesis']}
        combined_sents = copy.deepcopy(sents)
        combined_sents.update(all_hyps)

        sents_keys = list(sents.keys())
        all_hyps_keys = list(all_hyps.keys())
        
        # Create all_sents_keys and sent_dict
        all_sents_keys = sents_keys + all_hyps_keys
        sent_dict = combined_sents
        
        # Get children and parents based on overcomplete setting
        if self.config['overcomplete']:
            children = {}
            parents = defaultdict(list)
            for ki in range(len(all_hyps_keys)):
                children[all_hyps_keys[ki]] = sents_keys + all_hyps_keys[:ki]
                for child in children[all_hyps_keys[ki]]:
                    parents[child].append(all_hyps_keys[ki])
            
            # Overcomplete tree
            ent_keys = [{'premises': sents_keys + all_hyps_keys[:ki], 
                        'hypothesis': all_hyps_keys[ki]} for ki in range(len(all_hyps_keys))]
        else:
            # Non-overcomplete tree
            children, parents = self.parse_lisp_proof(data_example['meta']['lisp_proof'])
            ent_keys = []
            for conclusion, premises in children.items():
                ent_keys.append({
                    'premises': premises,
                    'hypothesis': conclusion
                })
        
        # Create premises_idxs and hypothesis_idxs
        premises_idxs = []
        hypothesis_idxs = []
        
        for ent in ent_keys:
            premise_indices = [all_sents_keys.index(p) for p in ent['premises']]
            hypothesis_index = all_sents_keys.index(ent['hypothesis'])
            premises_idxs.append(premise_indices)
            hypothesis_idxs.append(hypothesis_index)
        
        # Create inputs
        inputs = {
            'premises': [],
            'hypothesis': []
        }
        
        for i in range(len(all_sents_keys)):
            # For each sentence, create a list of premises (empty for leaf nodes)
            premises_for_sent = []
            for ent_idx, hyp_idx in enumerate(hypothesis_idxs):
                if hyp_idx == i:
                    premises_for_sent = [sent_dict[all_sents_keys[p_idx]] for p_idx in premises_idxs[ent_idx]]
                    break
            inputs['premises'].append(premises_for_sent)
            inputs['hypothesis'].append(sent_dict[all_sents_keys[i]])
        
        # ent_inputs = [{
        #     'premises': [f"{premise}: {sent_dict[premise]}" for premise in d['premises']], 
        #     'hypothesis': f"{d['hypothesis']}: {sent_dict[d['hypothesis']]}"
        # } for d in ent_keys]

        # overcomplete tree
        # always use overcomplete tree even when we are doing non-overcomplete tree to give an ordering
        ent_inputs = [{
            'premises': [f"{premise}: {combined_sents[premise]}" for premise in d['premises']], 
            'hypothesis': f"{d['hypothesis']}: {all_hyps[d['hypothesis']]}"
        } for d in ent_keys]

        return {
            'idx': idx,
            'all_sents_keys': all_sents_keys,
            'sent_dict': sent_dict,
            'premises_idxs': premises_idxs,
            'hypothesis_idxs': hypothesis_idxs,
            'inputs': inputs,
            'children': children,
            'parents': parents,
            'ent_inputs': ent_inputs
        }
    
    @staticmethod
    def parse_lisp_proof(lisp_proof):
        tokens = re.findall(r'[()]|[^ ()]+', lisp_proof)

        stack = []
        children = {}
        parents = {}

        for token in tokens:
            if token == '(':
                stack.append([])
            elif token == ')':
                completed = stack.pop()
                if '->' in completed:
                    arrow_index = completed.index('->')
                    premises = completed[:arrow_index]
                    conclusion = completed[arrow_index + 1]

                    children[conclusion] = premises
                    for premise in premises:
                        parents[premise] = conclusion

                    if stack:
                        stack[-1].append(conclusion)
                else:
                    if stack:
                        stack[-1].extend(completed)
            else:
                stack[-1].append(token)

        return children, parents

    
class EBDatasetReplaceGlitch(EBDataset):
    def __init__(self, config, split, root_dir):
        super().__init__(config, split, root_dir)
        self.data = self.get_raw_data()
        
        
    def __getitem__(self, idx):
        """
        Get a processed example from the dataset
        """
        data_example = self.data[idx]
        
        sents = data_example['meta']['triples']
        if 'intermediate_conclusions' in data_example['meta']:
            all_hyps = data_example['meta']['intermediate_conclusions']
        else:
            all_hyps = {'hypothesis': data_example['meta']['hypothesis']}
        
        # randomly replace one of the intermediate conclusions with a random fact from the delete list
        # Create a deterministic seed based on the context
        context_str = data_example['context']
        seed_value = sum(ord(c) for c in context_str)
        local_random = random.Random(seed_value)
        
        replace_key = local_random.choice(list(all_hyps.keys()))
        all_facts = [fact['fact'] for fact in data_example['meta']['delete_list']]
        all_facts = [fact for fact in all_facts if fact not in all_hyps.values()]
        replace_value = local_random.choice(all_facts)
        all_hyps[replace_key] = replace_value

        combined_sents = copy.deepcopy(sents)
        combined_sents.update(all_hyps)

        sents_keys = list(sents.keys())
        all_hyps_keys = list(all_hyps.keys())
        
        # Create all_sents_keys and sent_dict
        all_sents_keys = sents_keys + all_hyps_keys
        sent_dict = combined_sents
        
        # Get children and parents based on overcomplete setting
        if self.config['overcomplete']:
            children = {}
            parents = defaultdict(list)
            for ki in range(len(all_hyps_keys)):
                children[all_hyps_keys[ki]] = sents_keys + all_hyps_keys[:ki]
                for child in children[all_hyps_keys[ki]]:
                    parents[child].append(all_hyps_keys[ki])
            
            # Overcomplete tree
            ent_keys = [{'premises': sents_keys + all_hyps_keys[:ki], 
                        'hypothesis': all_hyps_keys[ki]} for ki in range(len(all_hyps_keys))]
        else:
            # Non-overcomplete tree
            children, parents = self.parse_lisp_proof(data_example['meta']['lisp_proof'])
            ent_keys = []
            for conclusion, premises in children.items():
                ent_keys.append({
                    'premises': premises,
                    'hypothesis': conclusion
                })

        # Non-overcomplete tree
        children_non_overcomplete, parents_non_overcomplete = self.parse_lisp_proof(data_example['meta']['lisp_proof'])
        ent_keys_non_overcomplete = []
        for conclusion, premises in children_non_overcomplete.items():
            ent_keys_non_overcomplete.append({
                'premises': premises,
                'hypothesis': conclusion
            })

        # # only put the direct parents in the impacted claims
        # impacted_claims = set(parents[replace_key])
        # import pdb; pdb.set_trace()

        # Find all claims impacted by the replaced claim
        impacted_claims = set([replace_key])
        queue = [replace_key]
        
        # Traverse the tree upwards from the replaced claim
        while queue:
            current_claim = queue.pop(0)
            if current_claim in parents_non_overcomplete:
                parent = parents_non_overcomplete[current_claim]

                if parent not in impacted_claims:
                    impacted_claims.add(parent)
                    queue.append(parent)
        
        # # Convert to list for easier use later
        impacted_claims = list(impacted_claims)
        
        # Create premises_idxs and hypothesis_idxs
        premises_idxs = []
        hypothesis_idxs = []
        
        for ent in ent_keys:
            premise_indices = [all_sents_keys.index(p) for p in ent['premises']]
            hypothesis_index = all_sents_keys.index(ent['hypothesis'])
            premises_idxs.append(premise_indices)
            hypothesis_idxs.append(hypothesis_index)
        
        # Create inputs
        inputs = {
            'premises': [],
            'hypothesis': []
        }
        
        for i in range(len(all_sents_keys)):
            # For each sentence, create a list of premises (empty for leaf nodes)
            premises_for_sent = []
            for ent_idx, hyp_idx in enumerate(hypothesis_idxs):
                if hyp_idx == i:
                    premises_for_sent = [sent_dict[all_sents_keys[p_idx]] for p_idx in premises_idxs[ent_idx]]
                    break
            inputs['premises'].append(premises_for_sent)
            inputs['hypothesis'].append(sent_dict[all_sents_keys[i]])
        
        # ent_inputs = [{
        #     'premises': [f"{premise}: {sent_dict[premise]}" for premise in d['premises']], 
        #     'hypothesis': f"{d['hypothesis']}: {sent_dict[d['hypothesis']]}"
        # } for d in ent_keys]

        # overcomplete tree
        # always use overcomplete tree even when we are doing non-overcomplete tree to give an ordering
        ent_inputs = [{
            'premises': [f"{premise}: {combined_sents[premise]}" for premise in d['premises']], 
            'hypothesis': f"{d['hypothesis']}: {all_hyps[d['hypothesis']]}"
        } for d in ent_keys]

        return {
            'idx': idx,
            'all_sents_keys': all_sents_keys,
            'sent_dict': sent_dict,
            'premises_idxs': premises_idxs,
            'hypothesis_idxs': hypothesis_idxs,
            'inputs': inputs,
            'children': children,
            'parents': parents,
            'ent_inputs': ent_inputs,
            'replace_key': replace_key,
            'replace_value': replace_value,
            'impacted_claims': impacted_claims
        }

class EBDatasetReplaceRawGlitch(EBDataset):
    def __init__(self, config, split, root_dir):
        super().__init__(config, split, root_dir)
        self.data = self.get_raw_data()
        
        
    def __getitem__(self, idx):
        """
        Get a processed example from the dataset
        """
        data_example = self.data[idx]
        
        sents = data_example['meta']['triples']
        if 'intermediate_conclusions' in data_example['meta']:
            all_hyps = data_example['meta']['intermediate_conclusions']
        else:
            all_hyps = {'hypothesis': data_example['meta']['hypothesis']}
        
        # randomly replace one of the intermediate conclusions with a random fact from the delete list
        # Create a deterministic seed based on the context
        context_str = data_example['context']
        seed_value = sum(ord(c) for c in context_str)
        local_random = random.Random(seed_value)
        
        # Randomly replace a raw claim (from triples) instead of an intermediate conclusion
        replace_key = local_random.choice(list(sents.keys()))
        all_facts = [fact['fact'] for fact in data_example['meta']['delete_list']]
        all_facts = [fact for fact in all_facts if fact not in sents.values() and fact not in all_hyps.values()]
        replace_value = local_random.choice(all_facts)
        
        sents[replace_key] = replace_value

        combined_sents = copy.deepcopy(sents)
        combined_sents.update(all_hyps)

        sents_keys = list(sents.keys())
        all_hyps_keys = list(all_hyps.keys())
        
        # Create all_sents_keys and sent_dict
        all_sents_keys = sents_keys + all_hyps_keys
        sent_dict = combined_sents
        
        # Get children and parents based on overcomplete setting
        if self.config['overcomplete']:
            children = {}
            parents = defaultdict(list)
            for ki in range(len(all_hyps_keys)):
                children[all_hyps_keys[ki]] = sents_keys + all_hyps_keys[:ki]
                for child in children[all_hyps_keys[ki]]:
                    parents[child].append(all_hyps_keys[ki])
            
            # Overcomplete tree
            ent_keys = [{'premises': sents_keys + all_hyps_keys[:ki], 
                        'hypothesis': all_hyps_keys[ki]} for ki in range(len(all_hyps_keys))]
        else:
            # Non-overcomplete tree
            children, parents = self.parse_lisp_proof(data_example['meta']['lisp_proof'])
            ent_keys = []
            for conclusion, premises in children.items():
                ent_keys.append({
                    'premises': premises,
                    'hypothesis': conclusion
                })

        # Non-overcomplete tree
        children_non_overcomplete, parents_non_overcomplete = self.parse_lisp_proof(data_example['meta']['lisp_proof'])
        ent_keys_non_overcomplete = []
        for conclusion, premises in children_non_overcomplete.items():
            ent_keys_non_overcomplete.append({
                'premises': premises,
                'hypothesis': conclusion
            })

        # # only put the direct parents in the impacted claims
        # impacted_claims = set(parents[replace_key])
        # import pdb; pdb.set_trace()

        # Find all claims impacted by the replaced claim
        impacted_claims = set([replace_key])
        queue = [replace_key]
        
        # Traverse the tree upwards from the replaced claim
        while queue:
            current_claim = queue.pop(0)
            if current_claim in parents_non_overcomplete:
                parent = parents_non_overcomplete[current_claim]

                if parent not in impacted_claims:
                    impacted_claims.add(parent)
                    queue.append(parent)
        
        # # Convert to list for easier use later
        impacted_claims = list(impacted_claims)
        # Create premises_idxs and hypothesis_idxs
        premises_idxs = []
        hypothesis_idxs = []
        
        for ent in ent_keys:
            premise_indices = [all_sents_keys.index(p) for p in ent['premises']]
            hypothesis_index = all_sents_keys.index(ent['hypothesis'])
            premises_idxs.append(premise_indices)
            hypothesis_idxs.append(hypothesis_index)
        
        # Create inputs
        inputs = {
            'premises': [],
            'hypothesis': []
        }
        
        for i in range(len(all_sents_keys)):
            # For each sentence, create a list of premises (empty for leaf nodes)
            premises_for_sent = []
            for ent_idx, hyp_idx in enumerate(hypothesis_idxs):
                if hyp_idx == i:
                    premises_for_sent = [sent_dict[all_sents_keys[p_idx]] for p_idx in premises_idxs[ent_idx]]
                    break
            inputs['premises'].append(premises_for_sent)
            inputs['hypothesis'].append(sent_dict[all_sents_keys[i]])
        
        # ent_inputs = [{
        #     'premises': [f"{premise}: {sent_dict[premise]}" for premise in d['premises']], 
        #     'hypothesis': f"{d['hypothesis']}: {sent_dict[d['hypothesis']]}"
        # } for d in ent_keys]

        # overcomplete tree
        # always use overcomplete tree even when we are doing non-overcomplete tree to give an ordering
        ent_inputs = [{
            'premises': [f"{premise}: {combined_sents[premise]}" for premise in d['premises']], 
            'hypothesis': f"{d['hypothesis']}: {all_hyps[d['hypothesis']]}"
        } for d in ent_keys]

        return {
            'idx': idx,
            'all_sents_keys': all_sents_keys,
            'sent_dict': sent_dict,
            'premises_idxs': premises_idxs,
            'hypothesis_idxs': hypothesis_idxs,
            'inputs': inputs,
            'children': children,
            'parents': parents,
            'ent_inputs': ent_inputs,
            'replace_key': replace_key,
            'replace_value': replace_value,
            'impacted_claims': impacted_claims
        }
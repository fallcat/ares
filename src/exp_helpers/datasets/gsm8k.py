from .base import BaseDataset
import os
import json
import copy
from collections import defaultdict
from nltk import sent_tokenize
import nltk
nltk.download('punkt')
import math

from datasets import load_dataset

class GSM8KDataset(BaseDataset):
    def __init__(self, config, split, root_dir=None):
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
        if self.root_dir is None:
            data = load_dataset("openai/gsm8k", "main")[self.split]
        else:
            raise NotImplementedError("Just use the huggingface dataset for now")
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

        # only need to modify this part if you are creating the overcomplete tree with cot.
        # put all premise in sents, and intermediate conclusions in all_hyps
        sents = {f'sent{si + 1}': sent for si, sent in enumerate(sent_tokenize(data_example['question']))}
        all_hyps_sents = [sent.strip() for sent in data_example['answer'].split('\n')] 
        all_hyps_sents[-1] = "The answer is " + all_hyps_sents[-1].replace('#### ', '') + "."
        all_hyps = {f'int{si + 1}': sent for si, sent in enumerate(all_hyps_sents)}
        
        combined_sents = copy.deepcopy(sents)
        combined_sents.update(all_hyps)

        sents_keys = list(sents.keys())
        all_hyps_keys = list(all_hyps.keys())
        
        # Create all_sents_keys and sent_dict
        all_sents_keys = sents_keys + all_hyps_keys
        sent_dict = combined_sents
        
        # Get children and parents based on overcomplete setting
        # always use overcomplete tree
        children = {}
        parents = defaultdict(list)
        for ki in range(len(all_hyps_keys)):
            children[all_hyps_keys[ki]] = sents_keys + all_hyps_keys[:ki]
            for child in children[all_hyps_keys[ki]]:
                parents[child].append(all_hyps_keys[ki])
        
        # Overcomplete tree
        ent_keys = [{'premises': sents_keys + all_hyps_keys[:ki], 
                    'hypothesis': all_hyps_keys[ki]} for ki in range(len(all_hyps_keys))]
        
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
from .base import BaseDataset
import os
import json
import copy
from collections import defaultdict
from nltk import sent_tokenize
import nltk
nltk.download('punkt')
import math
from tqdm.auto import tqdm


class InliDataset(BaseDataset):
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
        data_dir = os.path.join(self.root_dir, self.config['data_dir'], split)

        # Sort files numerically by extracting the number from the filename
        files = sorted(os.listdir(data_dir), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf'))
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                    data.append(json.load(f))

        if self.config['valid_only']:
            new_data = []
            for data_example in data:
                derived_claims = ['.'.join(sent.split('.')[1:]).strip() for sent in data_example['cot'][self.config['task_id']].split('\n')]
                step_types = ['. '.join(sent.split('.')[1:]).strip() for sent in data_example['step_types'].split('\n')]
                if len(derived_claims) == len(step_types):
                    new_data.append(data_example)
            data = new_data

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

        raw_claims = sent_tokenize(data_example['premise'])
        derived_claims = ['.'.join(sent.split('.')[1:]).strip() for sent in data_example['cot'][self.config['task_id']].split('\n')]
        derived_claims.append(data_example[self.config['task_id']])
        step_types = ['. '.join(sent.split('.')[1:]).strip() for sent in data_example['step_types'].split('\n')]

        data_entry = self.get_data_entry(raw_claims, derived_claims)

        data_entry['step_types'] = step_types
        data_entry['raw_claims'] = raw_claims
        data_entry['derived_claims'] = derived_claims
        data_entry['task_id'] = self.config['task_id']
        data_entry['original_data'] = data_example

        return data_entry
        # # only need to modify this part if you are creating the overcomplete tree with cot.
        # # put all premise in sents, and intermediate conclusions in all_hyps
        # sents = {f'sent{si + 1}': sent for si, sent in enumerate(sent_tokenize(data_example['premise']))}
        # all_hyps_sents = ['.'.join(sent.split('.')[1:]).strip() for sent in data_example['cot'][self.config['task_id']].split('\n')] + [data_example[self.config['task_id']]]
        # all_hyps = {f'int{si + 1}': sent for si, sent in enumerate(all_hyps_sents)}
        
        # combined_sents = copy.deepcopy(sents)
        # combined_sents.update(all_hyps)

        # sents_keys = list(sents.keys())
        # all_hyps_keys = list(all_hyps.keys())
        
        # # Create all_sents_keys and sent_dict
        # all_sents_keys = sents_keys + all_hyps_keys
        # sent_dict = combined_sents
        
        # # Get children and parents based on overcomplete setting
        # # always use overcomplete tree
        # children = {}
        # parents = defaultdict(list)
        # for ki in range(len(all_hyps_keys)):
        #     children[all_hyps_keys[ki]] = sents_keys + all_hyps_keys[:ki]
        #     for child in children[all_hyps_keys[ki]]:
        #         parents[child].append(all_hyps_keys[ki])
        
        # # Overcomplete tree
        # ent_keys = [{'premises': sents_keys + all_hyps_keys[:ki], 
        #             'hypothesis': all_hyps_keys[ki]} for ki in range(len(all_hyps_keys))]
        
        # # Create premises_idxs and hypothesis_idxs
        # premises_idxs = []
        # hypothesis_idxs = []
        
        # for ent in ent_keys:
        #     premise_indices = [all_sents_keys.index(p) for p in ent['premises']]
        #     hypothesis_index = all_sents_keys.index(ent['hypothesis'])
        #     premises_idxs.append(premise_indices)
        #     hypothesis_idxs.append(hypothesis_index)
        
        # # Create inputs
        # inputs = {
        #     'premises': [],
        #     'hypothesis': []
        # }
        
        # for i in range(len(all_sents_keys)):
        #     # For each sentence, create a list of premises (empty for leaf nodes)
        #     premises_for_sent = []
        #     for ent_idx, hyp_idx in enumerate(hypothesis_idxs):
        #         if hyp_idx == i:
        #             premises_for_sent = [sent_dict[all_sents_keys[p_idx]] for p_idx in premises_idxs[ent_idx]]
        #             break
        #     inputs['premises'].append(premises_for_sent)
        #     inputs['hypothesis'].append(sent_dict[all_sents_keys[i]])

        # # overcomplete tree
        # # always use overcomplete tree even when we are doing non-overcomplete tree to give an ordering
        # ent_inputs = [{
        #     'premises': [f"{premise}: {combined_sents[premise]}" for premise in d['premises']], 
        #     'hypothesis': f"{d['hypothesis']}: {all_hyps[d['hypothesis']]}"
        # } for d in ent_keys]

        # return {
        #     'idx': idx,
        #     'all_sents_keys': all_sents_keys,
        #     'sent_dict': sent_dict,
        #     'premises_idxs': premises_idxs,
        #     'hypothesis_idxs': hypothesis_idxs,
        #     'inputs': inputs,
        #     'children': children,
        #     'parents': parents,
        #     'ent_inputs': ent_inputs
        # }

class InliDatasetInjectingCotGlitch(InliDataset):
    def __init__(self, config, split, root_dir):
        super().__init__(config, split, root_dir)
        self.data = self.get_raw_data()

    def __getitem__(self, idx):
        """
        Get a processed example from the dataset
        """
        data_example = self.data[idx]

        # only need to modify this part if you are creating the overcomplete tree with cot.
        # put all premise in sents, and intermediate conclusions in all_hyps
        sents = {f'sent{si + 1}': sent for si, sent in enumerate(sent_tokenize(data_example['premise']))}
        all_hyps_sents = ['.'.join(sent.split('.')[1:]).strip() for sent in data_example['cot'][self.config['task_id']].split('\n')] + [data_example[self.config['task_id']]]
        
        # Handle injections if they exist
        if 'injection' in data_example and self.config['task_id'] in data_example['injection']:
            injection_info = data_example['injection'][self.config['task_id']]
            if 'Injection place' in injection_info:
                # Insert the injection at the specified place and shift everything else
                injection_place = injection_info['Injection place'] - 1  # Convert to 0-indexed
                injection_text = injection_info['Injection']
                all_hyps_sents.insert(injection_place, injection_text)
            elif 'Replace place' in injection_info:
                # Replace the claim at the specified place
                replace_place = injection_info['Replace place'] - 1  # Convert to 0-indexed
                replacement_text = injection_info['Injection']
                if 0 <= replace_place < len(all_hyps_sents):
                    all_hyps_sents[replace_place] = replacement_text
        
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

        # {
        #     "idx": 0,
        #     "dataset": "circa",
        #     "premise": "Lee and Coleman are colleagues who are leaving work on a Friday at the same time. Lee says, \"Are you spending time with friends or family over the weekend?\" Coleman responds, \"I have too much work to do.\"",
        #     "implied_entailment": "This weekend, Coleman is avoiding spending time with friends and family.",
        #     "explicit_entailment": "Coleman has too much work to do over the weekend.",
        #     "neutral": "Coleman will stay home this weekend.",
        #     "contradiction": "This weekend, Coleman is with friends and family.",
        #     "cot": {
        #         "implied_entailment": "1. Lee and Coleman are colleagues who are leaving work on a Friday at the same time.  \n2. Lee inquires about Coleman's weekend plans, specifically asking if he will spend time with friends or family.  \n3. Coleman responds by stating that he has too much work to do.  \n4. Coleman's response implies that he will be occupied with work over the weekend.  \n5. By stating he has too much work, Coleman suggests that he will not have time for other activities.  \n6. Spending time with friends or family is an activity that requires time.  \n7. Since Coleman indicates he will be busy with work, it implies he will not engage in other activities.  \n8. Therefore, Coleman is avoiding spending time with friends and family this weekend.",
        #         "explicit_entailment": "1. Lee and Coleman are colleagues who are leaving work on a Friday at the same time.  \n2. Lee inquires about Coleman's weekend plans, specifically asking if he will spend time with friends or family.  \n3. Coleman's response to Lee's question is, \"I have too much work to do.\"  \n4. Coleman's response indicates that he has work obligations that will occupy his time over the weekend.  \n5. Therefore, it can be concluded that Coleman has too much work to do over the weekend.",
        #         "neutral": "1. Lee and Coleman are colleagues who are leaving work on a Friday at the same time, indicating the end of the workweek.  \n2. Lee inquires about Coleman's weekend plans, suggesting that weekends are typically a time for leisure activities such as spending time with friends or family.  \n3. Coleman responds by stating, \"I have too much work to do,\" implying that he has work obligations that extend into the weekend.  \n4. The statement \"I have too much work to do\" suggests that Coleman prioritizes his work responsibilities over social activities.  \n5. If Coleman has too much work to do, it is likely that he will need to dedicate his weekend to completing these tasks.  \n6. Completing work tasks typically requires a conducive environment, which is often found at home.  \n7. Therefore, it is reasonable to conclude that Coleman will stay home this weekend to focus on his work.",
        #         "contradiction": "1. Lee and Coleman are colleagues who are leaving work on a Friday at the same time.  \n2. Lee inquires about Coleman's weekend plans, specifically asking if he will spend time with friends or family.  \n3. Coleman responds by saying he has too much work to do, implying he plans to work over the weekend.  \n4. People often say they have work to do as a polite way to decline social invitations or to avoid discussing personal plans.  \n5. Coleman might be using the excuse of having too much work to avoid discussing his actual weekend plans.  \n6. It is possible that Coleman values his privacy and prefers not to disclose his personal plans to Lee.  \n7. Given the social nature of weekends, it is common for people to spend time with friends and family.  \n8. Despite Coleman's statement about work, he might have decided to prioritize personal relationships over work commitments.  \n9. Therefore, it is concluded that Coleman is actually spending the weekend with friends and family."
        #     },
        #     "injection": {
        #         "implied_entailment": {
        #             "Injection": "Friday is often considered the end of the workweek in many cultures.",
        #             "Injection place": 2,
        #             "Injection type": "CORRECT_IRRELEVANT_TO_PREMISE"
        #         },
        #         "explicit_entailment": {
        #             "Injection": "Friday is often considered the end of the workweek in many cultures.",
        #             "Injection place": 2,
        #             "Injection type": "CORRECT_IRRELEVANT_TO_PREMISE"
        #         },
        #         "neutral": {
        #             "Injection": "Lee and Coleman work in a company that has a flexible work-from-home policy.",
        #             "Injection place": 3,
        #             "Injection type": "CORRECT_IRRELEVANT_TO_PREMISE"
        #         },
        #         "contradiction": {
        #             "Injection": "Lee and Coleman work in a company that has a flexible work-from-home policy.",
        #             "Injection place": 2,
        #             "Injection type": "CORRECT_IRRELEVANT_TO_PREMISE"
        #         }
        #     }
        # }

class InliDatasetInjectingWorldKnowledge(InliDataset):
    def __init__(self, config, split, root_dir):
        super().__init__(config, split, root_dir)
        self.data = self.get_raw_data()

    def get_world_knowledge(self):
        with open(os.path.join(self.root_dir, self.config['world_knowledge_file']), 'r', encoding='utf-8') as f:
            world_knowledge = json.load(f)
        return world_knowledge
    
    def get_raw_data(self):
        data = super().get_raw_data()
        self.world_knowledge = self.get_world_knowledge()
        self.world_knowledge_id = self.config.get('world_knowledge_id', 0)
        self.insert_region = self.config.get('insert_region', 'front')
        
        # Select the specific world knowledge claim based on the ID
        selected_knowledge = self.world_knowledge[self.world_knowledge_id % len(self.world_knowledge)]
        
        # Store the selected knowledge details as dataset properties
        self.world_knowledge_claim = selected_knowledge['claim']
        self.world_knowledge_known = selected_knowledge['known']
        
        return data

    def __getitem__(self, idx):
        """
        Get a processed example from the dataset
        """
        data_example = self.data[idx]
        # only need to modify this part if you are creating the overcomplete tree with cot.
        # put all premise in sents, and intermediate conclusions in all_hyps
        sents = {f'sent{si + 1}': sent for si, sent in enumerate(sent_tokenize(data_example['premise']))}
        # Get the reasoning trace and conclusion
        reasoning_trace = ['.'.join(sent.split('.')[1:]).strip() for sent in data_example['cot'][self.config['task_id']].split('\n')]
        conclusion = [data_example[self.config['task_id']]]
        
        # Inject world knowledge based on insert_region parameter
        if self.insert_region == 'front':
            # Insert at the beginning of the reasoning trace
            all_hyps_sents = reasoning_trace[:0] + [self.world_knowledge_claim] + reasoning_trace[0:] + conclusion
        elif self.insert_region == 'middle':
            # Insert in the middle of the reasoning trace
            mid_point = math.ceil(len(reasoning_trace) / 2)
            all_hyps_sents = reasoning_trace[:mid_point] + [self.world_knowledge_claim] + reasoning_trace[mid_point:] + conclusion
        elif self.insert_region == 'back':
            # Insert at the end of the reasoning trace, before the conclusion
            all_hyps_sents = reasoning_trace + [self.world_knowledge_claim] + conclusion
        else:
            # Default case - no injection
            all_hyps_sents = reasoning_trace + conclusion
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

class InliDatasetLLMGeneration(InliDataset):
    def __init__(self, config, split, root_dir):
        super().__init__(config, split, root_dir)
        self.data = self.get_raw_data()

    def get_raw_data(self):
        data = super().get_raw_data()
        # only keep data that have the wrong answer
        answer_mapping = {
            'implied_entailment': 'ENTAILED',
            'explicit_entailment': 'ENTAILED',
            'neutral': 'NEUTRAL',
            'contradiction': 'CONTRADICTING'
        }
        new_data = []
        for data_example in tqdm(data):
            if data_example['cot_parsed'][self.config['task_id']]['answer'] != answer_mapping[self.config['task_id']]:
                new_data.append(data_example)
        return new_data

    def __getitem__(self, idx):
        """
        Get a processed example from the dataset
        """
        data_example = self.data[idx]

        # only need to modify this part if you are creating the overcomplete tree with cot.
        # put all premise in sents, and intermediate conclusions in all_hyps
        sents = {f'sent{si + 1}': sent for si, sent in enumerate(sent_tokenize(data_example['premise']))}
        all_hyps_sents = [sent.strip() for sent in data_example['cot_parsed'][self.config['task_id']]['reasoning']]
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

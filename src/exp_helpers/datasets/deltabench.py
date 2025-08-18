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
import pandas as pd


class DeltaBenchDataset(BaseDataset):
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
        import json

        data = []
        with open(os.path.join(self.root_dir, self.config['data_dir'], 'Deltabench_v1.jsonl'), "r", encoding="utf-8") as f:
            for line in f:
                # strip the trailing newline and parse JSON
                data.append(json.loads(line))

        if 'min_length' in self.config:
            data = [d for d in data if len(d['sections']) >= self.config['min_length']]

        if 'task_l1' in self.config:
            data = [d for d in data if d['task_l1'] == self.config['task_l1']]

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
        raw_claims = sent_tokenize(data_example['question'])

        if self.config['derived_claim_mode'] == 'sections':
            derived_claims = [data_example['sections'][i]['content'] \
                  for i in range(len(data_example['sections']))]
        else: # 'steps'
            derived_claims = data_example['long_cot'].split('\n\n')

        data_entry = self.get_data_entry(raw_claims, derived_claims)

        data_entry['original_data'] = data_example
        data_entry['raw_claims'] = raw_claims
        data_entry['derived_claims'] = derived_claims

        return data_entry

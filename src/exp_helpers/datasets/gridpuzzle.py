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


class GridPuzzleDataset(BaseDataset):
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
        data = pd.read_excel(os.path.join(self.root_dir, self.config['data_dir'], f'{self.config["task_id"]}.xlsx'))

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
        data_example = self.data.iloc[idx]

        if self.config['raw_claim_mode'] == 'question':
            raw_claims = [data_example['question']]
        elif self.config['raw_claim_mode'] == 'question_line_split':
            raw_claims = [line.strip() for line in data_example['question'].split('\n') if line.strip() != '']
        elif self.config['raw_claim_mode'] == 'context_clue_only':
            curr = 'context'
            raw_claims = []
            context = ''
            for line in data_example['question'].split('\n'):
                if len(line.strip()) == 0:
                    continue
                if line.strip().lower().startswith('clue'):
                    curr = 'clue'
                    raw_claims.append(context)
                elif 'following format' in line.lower():
                    curr = 'format'
                    if curr == 'format':
                        break
                if curr == 'context':
                    context = context + line
                    # raw_claims.append(line.strip())
                elif curr == 'clue':
                    raw_claims.append(line.strip())
        else:
            raise ValueError(f"Invalid raw claim mode: {self.config['raw_claim_mode']}")

        annotated_rc = json.loads(data_example['annotated_RC'])
        derived_claims = [item['Sentence'] for item in annotated_rc]

        data_entry = self.get_data_entry(raw_claims, derived_claims)

        data_entry['derived_claims_annotated_rc'] = annotated_rc

        return data_entry

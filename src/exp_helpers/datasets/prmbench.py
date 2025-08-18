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
from datasets import load_dataset, Features, Sequence, Value


class PRMBenchDataset(BaseDataset):
    def __init__(self, config, split):
        """
        Initialize the dataset
        """
        super().__init__(config)
        self.split = split
        self.data = self.get_raw_data()

    def get_raw_data(self):
        """
        Get the raw data from the dataset
        """
        # --- 1.  Declare the full schema ------------------------------------------
        schema = Features({
            "original_question":  Value("string"),
            "modified_question":  Value("string"),
            "original_process":   Sequence(Value("string")),
            "modified_process":   Sequence(Value("string")),
            "modified_steps":     Sequence(Value("int64")),
            "error_steps":        Sequence(Value("int64")),
            "reason":             Value("string"),
            "idx":                Value("string"),
            "question":           Value("string"),
            "classification":     Value("string"),
            # ↓↓↓  the two columns that are sometimes missing  ↓↓↓
            "ground_truth":       Value("string"),      # optional – will be "" if absent
            "original_response":  Value("string"),      # optional – will be "" if absent
        })

        # --- 2.  Load the dataset with that schema ---------------------------------
        data = load_dataset(
                "hitsmy/PRMBench_Preview",
                split="train",
                features=schema,          # ← patch applied here
                # ignore_verifications=True # speeds things up, optional
        )

        if 'min_length' in self.config and self.config['min_length'] > 0:
            data = data.filter(lambda x: len(x['modified_process']) >= self.config['min_length'])

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

        # Determine which version to use based on task_id config
        task_id = self.config.get('task_id', 'modified')  # Default to modified for backward compatibility
        
        if task_id == 'original':
            # Use original question and process
            raw_claims = sent_tokenize(data_example['original_question'])
            derived_claims = data_example['original_process']
        else:
            # Use modified question and process (default behavior)
            raw_claims = sent_tokenize(data_example['modified_question'])
            derived_claims = data_example['modified_process']

        data_entry = self.get_data_entry(raw_claims, derived_claims)

        data_entry['original_data'] = data_example

        return data_entry

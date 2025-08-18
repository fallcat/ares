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
from datasets import load_dataset
import pandas as pd


class RevealDataset(BaseDataset):
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
        # 1) load & clean just as before
        reveal = load_dataset("google/reveal")
        data = reveal['eval'] if self.config['task_id']=='eval' else reveal['open']
        df = pd.DataFrame(data)

        df['step_idx'] = df['step_idx'].astype(int)
        df = df.sort_values(['answer_id','step_idx'])

        # 2) combine all evidence strings for each answer_id into one column
        df['evidence_all'] = (
            df
            .groupby('answer_id')['evidence']
            .transform(lambda evs: " || ".join(pd.unique(evs.dropna())))
        )

        # 3) pivot on the new “evidence_all” instead of the per-step “evidence”
        deduped = df.drop_duplicates(subset=['answer_id','step_idx'], keep='first')
        wide = deduped.pivot_table(
            index=['answer_id','question','full_answer','evidence_all'],
            columns='step_idx',
            values=['step','correctness_label'],
            aggfunc='first'
        )

        # 4) flatten the MultiIndex columns
        wide.columns = [
            f'{("step_text" if val=="step" else "correctness_label")}_step_{i}'
            for val,i in wide.columns
        ]
        wide = wide.reset_index()

        return wide


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

        question = data_example['question']
        if self.config['use_evidence']:
            evidence = data_example['evidence_all']
            raw_claims = sent_tokenize(evidence) + [question]
        else:
            raw_claims = [question]
        step_text_columns = sorted([key for key in data_example.keys() if 'step_text' in key], key=lambda x: int(x.split('_')[-1]))
        derived_claims = [data_example[column] for column in step_text_columns if pd.notna(data_example[column])]
        correctness_label_columns = sorted([key for key in data_example.keys() if 'correctness_label_' in key], key=lambda x: int(x.split('_')[-1]))
        correctness_labels = [data_example[column] for column in correctness_label_columns][:len(derived_claims)]

        # if use_attributes is True, use steps that have correctness label None as raw claims
        if self.config['use_attributes']:
            raw_claims = []
            for step_idx, label in enumerate(correctness_labels):
                if label is None:
                    raw_claims.append(derived_claims[step_idx])
            raw_claims = raw_claims + [question]

        data_entry = self.get_data_entry(raw_claims, derived_claims)
        data_entry['correctness_labels'] = correctness_labels

        return data_entry
import copy
import json
import logging
import random
from dataclasses import dataclass
from typing import Dict, Sequence

from transformers import AutoTokenizer
import numpy as np
import torch

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_train_val_dataset(train_path, valid_path=None):
    f = open(train_path, "r", encoding="utf-8")
    data = []
    while True:
        line = f.readline()
        if not line:
            break
        data.append(json.loads(line))
    f.close()
    train_data = []
    valid_data = []
    if valid_path:
        f = open(valid_path, "r", encoding="utf-8")
        while True:
            line = f.readline()
            if not line:
                break
            valid_data.append(json.loads(line))
        f.close()
        train_data = data
    else:
        train_data = data[10000:]
        valid_data = data[:10000]
    return train_data, valid_data


class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, nsamples, seqlen=1024):
        raw_data = dataset
        self.tokenizer = tokenizer
        self.seqlen = seqlen
        # tokenized_datasets = []
        # for i_data in raw_data:
        #     print(i_data["text"])
        #     trainenc = tokenizer(i_data["text"], return_tensors="pt")
        #     # print(trainenc)
        #     if trainenc.input_ids.shape[1] > block_size:
        #         print(f'more than {seqlen}: {trainenc.input_ids.shape[1]}')
        #         break
        #     trainloader.append(trainenc.input_ids)

        # for d in raw_data:
        #     tokenized_datasets.append(self.tokenize_function(d))
        # return self.tokenizer(examples["text"], return_tensors="pt")

        full_str = "\n\n".join(data['text'] for data in raw_data)
        # full_str = full_str.replace('<unk>', '')
        trainenc = tokenizer(full_str, return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            # tar = inp.clone()
            # tar[:, :-1] = -100
            trainloader.append(inp)
        self.input_ids = trainloader

        # grouped_dataset = self.group_texts(tokenized_datasets)
        # self.input_ids = grouped_dataset["input_ids"]
        # self.labels = grouped_dataset["labels"]
        # self.data = [
        #     dict(input_ids=self.input_ids[i], labels=self.labels[i])
        #     for i in range(len(self.input_ids))
        # ]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return self.input_ids[i]

    def __iter__(self):
        return iter(self.input_ids)

    # def tokenize_function(self, examples):
    #     return self.tokenizer(examples["text"], return_tensors="pt")

    # def group_texts(self, examples):
    #     # Concatenate all texts.
    #     # Initialize an empty dictionary
    #     concatenated_examples = {}

    #     # Loop through the list of dictionaries
    #     for d in examples:
    #         # Loop through the keys in each dictionary
    #         for key in d.keys():
    #             # If the key is not already a key in the dict_of_lists, create a new list
    #             if key not in concatenated_examples:
    #                 concatenated_examples[key] = []
    #             # Append the value to the list associated with the key in dict_of_lists
    #             concatenated_examples[key].extend(d[key])
    #     total_length = len(concatenated_examples["input_ids"])
    #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    #     # customize this part to your needs.
    #     if total_length >= self.block_size:
    #         total_length = (total_length // self.block_size) * self.block_size
    #     # Split by chunks of max_len.
    #     result = {
    #         k: [
    #             t[i : i + self.block_size]
    #             for i in range(0, total_length, self.block_size)
    #         ]
    #         for k, t in concatenated_examples.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()
    #     return result


def jload(filename, mode="r"):
    """Load a .json file into a dictionary."""
    with open(filename, mode) as f:
        jdict = json.load(f)
    return jdict

def get_train_val_dataset(train_path):
    f = open(train_path, "r", encoding="utf-8")
    data = []
    while True:
        line = f.readline()
        if not line:
            break
        data.append(json.loads(line))
    f.close()
    return data

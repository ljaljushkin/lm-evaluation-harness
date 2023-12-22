
from nncf.parameters import SensitivityMetric
import datetime
import gc
import shutil
import time
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable
import logging
import matplotlib.pyplot as plt
import numpy as np
import openvino.runtime as ov
from datasets import load_dataset
from openvino import Core
from optimum.intel import OVModelForCausalLM, OVQwenModel
from tqdm import tqdm
from transformers import AutoTokenizer
import os
from contextlib import redirect_stdout, redirect_stderr
from optimum.exporters import TasksManager
from optimum.utils import (
    NormalizedTextConfig, NormalizedConfigManager
)
from transformers import AutoConfig
from nncf import Dataset
from nncf import compress_weights
from nncf.parameters import CompressWeightsMode
core = Core()

def gen_pkv(num_heads, head_dim, num_layers=None):
    if num_layers is None:
        num_layers = num_heads
    res = {}
    for i in range(num_layers):
        res[f"past_key_values.{i}.key"] = np.zeros((1, num_heads, 0, head_dim))
        res[f"past_key_values.{i}.value"] = np.zeros((1, num_heads, 0, head_dim))
    return res

def gen_qwen_pkv(num_heads, head_dim, num_layers=None):
    if num_layers is None:
        num_layers = num_heads
    res = {}
    for i in range(num_layers):
        res[f"past_key_values.{i}.key"] = np.zeros((1, 0, num_heads, head_dim))
        res[f"past_key_values.{i}.value"] = np.zeros((1, 0, num_heads, head_dim))
    return res

def gen_pkv_bloom(num_heads, head_dim, num_layers=None):
    if num_layers is None:
        num_layers = num_heads
    res = {}
    for i in range(num_layers):
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        res[f"past_key_values.{i}.key"] = np.zeros((1 * num_heads, head_dim, 0))
        res[f"past_key_values.{i}.value"] = np.zeros((1 * num_heads, 0, head_dim))
    return res

def transform_func(item, tokenizer, gen_pkv_fn):
    tokens = tokenizer(item['text'])
    #return tokens['input_ids'], tokens['attention_mask']
    attention_mask = np.expand_dims(np.array(tokens['attention_mask']), 0)
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids = np.ma.array(position_ids, mask=attention_mask == 0)
    position_ids.filled(fill_value=1)
    res = {
        'input_ids': np.expand_dims(np.array(tokens['input_ids']), 0),
        'attention_mask': attention_mask,
        # 'position_ids': position_ids
    }
    res.update(gen_pkv_fn())
    return res

MODEL_IDS_VS_GEN_FN = {
    'facebook/opt-125m': partial(gen_pkv, 12, 64),
    'databricks/dolly-v2-3b': partial(gen_pkv, 32, 80),
    'meta-llama/Llama-2-7b-chat-hf': partial(gen_pkv, 32, 128),
    'meta-llama/Llama-2-7b-hf': partial(gen_pkv, 32, 128),
    'facebook/opt-6.7b': partial(gen_pkv, 32, 128),
    'bigscience/bloom-7b1': partial(gen_pkv_bloom, 32, 128, 30),
    'bigscience/bloomz-7b1': partial(gen_pkv_bloom, 32, 128, 30),
    'togethercomputer/RedPajama-INCITE-7B-Instruct': partial(gen_pkv, 32, 128),
    'meta-llama/Llama-2-13b-chat-hf': partial(gen_pkv, 40, 128),
    'databricks/dolly-v2-12b': partial(gen_pkv, 40, 128, 36),
    'openlm-research/open_llama_3b': None,
    'THUDM/chatglm2-6b': None,
    'THUDM/chatglm3-6b': None,
    'HuggingFaceH4/zephyr-7b-beta': partial(gen_pkv, 8, 128, 32),
    'bigscience/bloomz-560m': None,
    'EleutherAI/gpt-j-6b': None,
    'Qwen/Qwen-7B-Chat': partial(gen_qwen_pkv, 32, 128, 32),
    'stable-zephyr-3b-dpo': partial(gen_pkv, 32, 80),
    'stabilityai/stablelm-3b-4e1t': partial(gen_pkv, 32, 80),
}

@dataclass
class ExpDesc:
    model_id: str
    mode: CompressWeightsMode = CompressWeightsMode.INT4_SYM
    metric: SensitivityMetric = None
    ratio: float = 1
    group_size: int = 128
    is_revert: bool = False
    use_data: bool = False
    custom_tokenizer: str = None

    def __str__(self):
        return f'{self.model_id} ----> {self.get_exp_name()}'

    def get_compress_fn(self):
        if self.use_data:
            gen_pkv_fn = MODEL_IDS_VS_GEN_FN[self.model_id]
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            dataset = load_dataset('wikitext', 'wikitext-2-v1', split='train[:1000]')
            dataset = dataset.filter(lambda example: len(example["text"]) > 128)
            nncf_dataset = Dataset(dataset, partial(transform_func, tokenizer=tokenizer, gen_pkv_fn=gen_pkv_fn))
            result = partial(compress_weights, mode=self.mode, ratio=self.ratio, group_size=self.group_size, dataset=nncf_dataset, all_layers=False, sensitivity_metric=self.metric)
        else:
            result = partial(compress_weights, mode=self.mode, ratio=self.ratio, group_size=self.group_size)
        return result

    def get_exp_name(self):
        result = self.mode.value

        if self.group_size != -1:
            result += f'_g{self.group_size}'

        if self.ratio != 1:
            result += f'_r{self.ratio * 100:2.0f}'

        if self.use_data:
            result += '_' + self.metric.value
        return result

EXP_DESCS= [
    ExpDesc('facebook/opt-125m', mode=CompressWeightsMode.INT8, group_size=-1),
    ExpDesc('facebook/opt-125m', mode=CompressWeightsMode.INT4_ASYM, ratio=0.8, group_size=128, use_data=True, metric=SensitivityMetric.MAX_ACTIVATION_VARIANCE),
    ExpDesc('facebook/opt-125m', mode=CompressWeightsMode.INT4_ASYM, ratio=0.8, group_size=128, use_data=False, metric=SensitivityMetric.WEIGHT_QUANTIZATION_ERROR),
]

CACHE_DIR = Path('cache')
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir()

ov_name = 'openvino_model.xml'

print('All experiments summary:')
for desc in EXP_DESCS:
    print(desc)

for desc in tqdm(EXP_DESCS):
    model_id = desc.model_id
    exp_name = desc.get_exp_name()
    model_name = Path(model_id).name.lower()
    SRC_PATH = CACHE_DIR / model_name / 'fp16' / ov_name
    DST_PATH = CACHE_DIR / model_name / exp_name /  ov_name
    SRC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DST_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_filename = DST_PATH.parent / 'compress_weight.log'
    print('Log file: ', log_filename.resolve())
    with open(log_filename, 'w') as f, redirect_stdout(f), redirect_stderr(f):
        print(desc)
        print(SRC_PATH)
        print(DST_PATH)
        try:
            if not SRC_PATH.with_suffix('.bin').exists():
                use_pkv = True
                ov_model = OVModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    use_cache=use_pkv,
                    export=True
                )
                ov_model.save_pretrained(SRC_PATH.parent)
                ov_model._save_config(SRC_PATH.parent)
                fp32_model = ov_model.model
            else:
                fp32_model = core.read_model(model=SRC_PATH)
        except Exception as error:
            print("Reading FP32 model failed:", error)
            continue

        shutil.copyfile(SRC_PATH.parent / 'config.json', DST_PATH.parent / 'config.json')

        for file_to_copy in SRC_PATH.parent.glob('*token*'):
            shutil.copyfile(file_to_copy, DST_PATH.parent / file_to_copy.name)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.save_pretrained(DST_PATH.parent)

        try:
            start = time.time()
            model = desc.get_compress_fn()(fp32_model)
            print(f'compressing weights took {(time.time() - start):.1f} seconds')
            start = time.time()

            ov.save_model(model, DST_PATH, compress_to_fp16=False)
            print(f"saving model {DST_PATH} took {(time.time() - start):.1f} seconds")
        except Exception as error:
            print("Compression failed:", error)
            print(traceback.print_exc())
            continue

    del model
    gc.collect()

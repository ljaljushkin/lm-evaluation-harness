from functools import partial
import numpy as np
import nncf
from nncf import compress_weights, Dataset
from nncf.parameters import CompressWeightsMode

from nncf.experimental.common.compression import compression_compensation

from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from datasets import load_dataset
import openvino.runtime as ov
from openvino import Core
core = Core()

import shutil
import time
from pathlib import Path


def gen_pkv(num_heads, head_dim, num_layers=None):
    if num_layers is None:
        num_layers = num_heads
    res = {}
    for i in range(num_layers):
        res[f"past_key_values.{i}.key"] = np.zeros((1, num_heads, 0, head_dim))
        res[f"past_key_values.{i}.value"] = np.zeros((1, num_heads, 0, head_dim))
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


def transform_func(item, tokenizer):
    tokens = tokenizer(item['text'])
    #return tokens['input_ids'], tokens['attention_mask']

    res = {'input_ids': np.expand_dims(np.array(tokens['input_ids']), 0),
           'attention_mask': np.expand_dims(np.array(tokens['attention_mask']), 0)}

    # res.update(gen_pkv(12, 64)) # opt-125m "?,12,?,64"
    # res.update(gen_pkv(32, 128)) # redpajama, opt-6.7 "?,32,?,128"
    # res.update(gen_pkv(40, 128, 36)) # dolly-12b "?,40,?,128"
    res.update(gen_pkv_bloom(32, 128, 30)) # bloom-7b1,  "?,128,?"
    # res.update(gen_pkv_bloom(16, 64, 24)) # bloom-560m,  "?,64,?"

    return res



CACHE_DIR = Path('/home/devuser/nlyalyus/projects/lm-evaluation-harness/cache')
MODEL_ID = 'togethercomputer/RedPajama-INCITE-7B-Instruct'
MODEL_ID = 'facebook/opt-125m'
MODEL_ID = 'facebook/opt-6.7b'
MODEL_ID = 'bigscience/bloom-7b1'
# MODEL_ID = 'bigscience/bloom-560m'
# MODEL_ID = 'databricks/dolly-v2-12b'

MODEL_NAME = Path(MODEL_ID).name
TOKENIZER_NAME = MODEL_ID
SRC_PATH = CACHE_DIR / MODEL_NAME / 'fp32' / 'openvino_model.xml'

use_comprensation = True
ratio = 0.8

if use_comprensation:
    exp_name = f"nf4_g128_r80_data"
else:
    exp_name = f"nf4_g128_r80"
DST_PATH = CACHE_DIR / MODEL_NAME / exp_name / 'openvino_model.xml'
print(DST_PATH)

if not SRC_PATH.exists():
    use_pkv = True
    ov_model = OVModelForCausalLM.from_pretrained(MODEL_ID, use_cache=use_pkv, trust_remote_code=True, export=True)
    ov_model.save_pretrained(SRC_PATH.parent)
    ov_model._save_config(SRC_PATH.parent)
    fp32_model = ov_model.model
else:
    print(f'Reading model {str(SRC_PATH)}')
    fp32_model = core.read_model(model=SRC_PATH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

start = time.perf_counter()
dataset = load_dataset('wikitext', 'wikitext-2-v1', split='train[:1000]')
dataset = dataset.filter(lambda example: len(example["text"]) > 128)

nncf_dataset = Dataset(dataset, partial(transform_func, tokenizer=tokenizer))

compress_weights_fn = partial(compress_weights, mode=CompressWeightsMode.NF4, group_size=128, ratio=ratio)

print(f'Started weight compression!')
if use_comprensation:
    compressed_model = compression_compensation.compression_compensation(fp32_model, nncf_dataset, compress_weights_fn)
else:
    compressed_model = compress_weights_fn(fp32_model)

end = time.perf_counter()

print("Time: ", end - start)

ov.save_model(compressed_model, DST_PATH, compress_to_fp16=False)
shutil.copyfile(SRC_PATH.parent / 'config.json', DST_PATH.parent / 'config.json')

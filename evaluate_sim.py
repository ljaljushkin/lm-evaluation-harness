import os
from pathlib import Path
from typing import List
import traceback
from nncf import compress_weights
from tqdm import trange
import json
import shutil
import random
import copy
from optimum.intel.openvino import (
    OVModelForCausalLM,
    OVQwenModel
)
from transformers import (
    AutoTokenizer,
    AutoConfig
)
from contextlib import redirect_stdout, redirect_stderr
from nncf.parameters import CompressWeightsMode
from whowhatbench import Evaluator

from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
from optimum.exporters import TasksManager
from openvino import Core
core = Core()

TasksManager._SUPPORTED_MODEL_TYPE["stablelm-epoch"] = TasksManager._SUPPORTED_MODEL_TYPE["llama"]
NormalizedConfigManager._conf["stablelm-epoch"] = NormalizedTextConfig.with_args(
    num_layers="num_hidden_layers",
    num_attention_heads="num_attention_heads",
)
NormalizedConfigManager._conf['qwen'] = NormalizedTextConfig.with_args(
    num_layers='num_hidden_layers', num_attention_heads='num_attention_heads', hidden_size='hidden_size'
)

MODEL_NAMES = [
    'stable-zephyr-3b-dpo',
    # 'llama-2-7b-chat-hf',
    'stablelm-3b-4e1t',
    # 'zephyr-7b-beta',
    # 'qwen-7b-chat'
]
for MODEL_NAME in MODEL_NAMES:
    cache_dir = Path(os.readlink('/home/devuser/nlyalyus/projects/lm-evaluation-harness/cache'))
    ROOT_DIR = cache_dir / MODEL_NAME
    gold_folder = ROOT_DIR / "fp16"
    gold_ir_path = gold_folder / "openvino_model.xml"
    gold_csv = gold_folder / 'gold_all.csv'
    config_gold = AutoConfig.from_pretrained(gold_folder, trust_remote_code=True)
    # config_gold = AutoConfig.from_pretrained('Qwen/Qwen-7B-Chat', trust_remote_code=True)
    tokenizer_gold = AutoTokenizer.from_pretrained(gold_folder)
    # tokenizer_gold = AutoTokenizer.from_pretrained('Qwen/Qwen-7B-Chat')
    tokenizer_gold.save_pretrained(gold_folder)
    print('gold path:', gold_csv.resolve())
    if gold_csv.exists():
        evaluator = Evaluator(tokenizer=tokenizer_gold, gt_data=gold_csv, test_data=str(gold_csv))
    else:
        # model_gold = OVQwenModel.from_pretrained(gold_folder, config=config_gold, trust_remote_code=True, use_cache=True)
        model_gold = OVModelForCausalLM.from_pretrained(gold_folder, config=config_gold, trust_remote_code=True, use_cache=True)
        evaluator = Evaluator(base_model=model_gold, tokenizer=tokenizer_gold)
        # dataset = load_dataset('ceval/ceval-exam', 'high_school_geography', split='test')
        # prompts = list(dataset["question"])[:24]
        # evaluator = Evaluator(base_model=model_gold, tokenizer=tokenizer_gold, test_data=prompts,
        #                       similarity_model_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        evaluator.dump_gt(gold_csv)

    EXP_NAMES = [
        # 'int4_sym_g64_r80_greedy0_anti',
        # 'int4_sym_g64_r80_greedy0',
        # 'int4_sym_g64_r80_greedy1',

        'int4_sym_g64_r80',
        'int4_sym_g64_r80_hawq_in',
        'int4_sym_g64_r80_max_var',
        'int4_sym_g64_r80_mean_var',
        'int4_sym_g64_r80_mean_max',

        # 'int4_sym_g128_r80',
        # 'int4_sym_g128_r80_hawq_in',
        # 'int4_sym_g128_r80_max_var',
        # 'int4_sym_g128_r80_mean_var',
        # 'int4_sym_g128_r80_mean_max',

        # 'int4_sym_g64_r80_criteria_OUT2',
        # 'int4_sym_g64_r80_baseline',
        # 'int4_sym_g64_r80_mean_var',
        # 'int4_sym_g64_r80_max_var',
        # 'int4_sym_g64_r80_mean_max',
    ]

    for exp_name in EXP_NAMES:
        cmp_ir_folder = ROOT_DIR / exp_name
        cmp_ir_path = cmp_ir_folder / "openvino_model.xml"
        cmp_model = core.read_model(model=cmp_ir_path)
        # cmp_model = OVModelForCausalLM.from_pretrained(cmp_ir_folder, config=config, trust_remote_code=True, use_cache=True)
        cmp_model = OVModelForCausalLM(cmp_model, config=config_gold, trust_remote_code=True, use_cache=True, model_save_dir=cmp_ir_folder)
        all_metrics_per_question, all_metrics = evaluator.score(cmp_model)
        print(all_metrics)
        similarity = float(all_metrics['similarity'].iloc[0])
        sdt_norm = float(all_metrics['SDT norm'].iloc[0])
        score = (similarity + 1 - sdt_norm) / 2
        # print(all_metrics_per_question)
        print('final score=', score)
        all_metrics['weighted'] = [score]
        model_cache_dir = cmp_ir_folder / 'model_cache'
        if model_cache_dir.exists():
            shutil.rmtree(model_cache_dir)
        all_metrics.to_csv(cmp_ir_folder / 'eval.csv')
from pathlib import Path
from typing import List
import traceback
from nncf import compress_weights
from tqdm import trange
import json
import shutil
import random
import copy
from optimum.intel.openvino import OVModelForCausalLM
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

ROOT_DIR = Path('/home/devuser/nlyalyus/projects/lm-evaluation-harness/cache/stable-zephyr-3b-dpo')
gold_folder = ROOT_DIR / "fp16"
gold_ir_path = gold_folder / "openvino_model.xml"
config = AutoConfig.from_pretrained(gold_folder, trust_remote_code=True)
# model_gold = OVModelForCausalLM.from_pretrained(gold_folder, config=config, trust_remote_code=True, use_cache=True)
tokenizer_gold = AutoTokenizer.from_pretrained(gold_folder)
evaluator = Evaluator(tokenizer=tokenizer_gold, gt_data='gold_5.csv', test_data_path='gold_5.csv')
# evaluator = Evaluator(base_model=model_gold, tokenizer=tokenizer_gold)
# evaluator.dump_gt('gold.csv')

def save_progress(list_scores_per_stage, optimal_ids):
    scores_path = gold_folder / 'greedy_search_result_20.json'
    results = {
        'scores_per_stage': list_scores_per_stage,
        'optimal_ids': optimal_ids,
    }
    with scores_path.open('w') as f:
        json.dump(results, f, indent=2)


log_filename = gold_folder / 'greedy_search.log'
print('Log file: ', log_filename.resolve())
with open(log_filename, 'w') as f, redirect_stdout(f), redirect_stderr(f):
    try:
        num_internal_ids = 224
        list_scores_per_stage: List[List[int]] = []
        # TRY 20 optimal after first iteration
        optimal_ids = [96, 29, 30, 77, 41, 156, 88, 62, 53, 154, 84, 21, 22, 7, 19, 93, 12, 28, 138, 223]
        for stage in range(1):#num_internal_ids):
            metric_per_id = []
            for i in trange(num_internal_ids, desc='Exhaustive search for sensitive layer'):
                if i in optimal_ids:
                    metric_per_id.append(0)
                    continue
                gold_ov_model = core.read_model(model=gold_ir_path)
                force_int8_ids = optimal_ids + [i]
                print('evaluate config: ', force_int8_ids)
                compressed_model = compress_weights(gold_ov_model, mode=CompressWeightsMode.INT4_SYM, ratio=1, group_size=64, force_int8_ids=force_int8_ids)
                cmp_model = OVModelForCausalLM(compressed_model, config=config, trust_remote_code=True, use_cache=True, model_save_dir=gold_folder)
                all_metrics_per_question, all_metrics = evaluator.score(cmp_model)
                print(all_metrics)
                similarity = float(all_metrics['similarity'].iloc[0])
                sdt_norm = float(all_metrics['SDT norm'].iloc[0])
                score = (similarity + 1 - sdt_norm) / 2
                # print(all_metrics_per_question)
                metric_per_id.append(score)
                print('final score=', score)
                model_cache_dir = gold_folder / 'model_cache'
                if model_cache_dir.exists():
                    shutil.rmtree(model_cache_dir)
            best_score = max(metric_per_id)
            optimal_id = metric_per_id.index(best_score)
            print('all scores=', metric_per_id)
            print('best score=', best_score)
            print('choose layer with index=', optimal_id)
            optimal_ids.append(optimal_id)
            list_scores_per_stage.append(metric_per_id)
            save_progress(list_scores_per_stage, optimal_ids)

    except Exception as error:
        print(traceback.print_exc())
    finally:
        save_progress(list_scores_per_stage, optimal_ids)
        model_cache_dir = gold_folder / 'model_cache'
        if model_cache_dir.exists():
            shutil.rmtree(model_cache_dir)
import argparse
import gc
import json
import os
import traceback
from dataclasses import dataclass
from lm_eval import evaluator
import shutil
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
from time import time, sleep
import random
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM
from collections import OrderedDict
from lm_eval import evaluator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
logging.getLogger("openai").setLevel(logging.WARNING)
from pathlib import Path
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')

LOGS_DIR = Path("./logs_compress")


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", required=True)
    # parser.add_argument(
    #     "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    # )
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=100)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true", default=True)
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--delete_ir_cache", action="store_true", default=False)
    parser.add_argument("--do_eval", action="store_true", default=False)

    return parser.parse_args()

@dataclass
class ExpDesc:
    model_id: str
    group_size: int = 64
    mode: str ='nf4'
    limit: float = None
    is_mixed: bool = False
    do_eval: bool = True
    delete_ir_cache: bool = False
    is_fp32: bool = False
    exp_name: str = None
    is_bin_needed: bool = False

    def get_encoded_name(self):
        if self.is_fp32:
            return 'fp32'
        if self.exp_name:
            return self.exp_name
        group_str = f'_g{self.group_size}' if self.group_size >= 2 else ''
        mixed_str = '_mixed' if self.is_mixed else ''
        return f'{self.mode}{group_str}{mixed_str}'

def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)


    all_results_paths = []
    # desc = ExpDesc("mistralai/Mixtral-8x7B-Instruct-v0.1", exp_name='fp16')
    desc = ExpDesc("dfurman/Mixtral-8x7B-Instruct-v0.1", exp_name='fp16')
    # desc = ExpDesc("mistralai/Mixtral-8x7B-v0.1", exp_name='fp16')

    model_id = desc.model_id
    model_name = model_id.replace('/', '__')
    model_args = f'pretrained={model_id}'

    total_num_experts = 8
    # exp_name = 'perlayer_alpha'
    # exp_name = 'perlayer_hitrate'
    exp_name = 'glb_thr_alpha'
    # exp_name = 'glb_thr_alpha_clipped'
    # exp_name = 'glb_thr_hitrate'
    # exp_name = 'glb_thr_alpha_trace'
    # exp_name = 'glb_thr_hitrate_trace'
    # exp_name = 'glb_thr_hitrate_25trace'
    # exp_name = 'glb_thr_alpha_25max'
    # exp_name = 'glb_thr_alpha_25max_clip5'
    # exp_name = 'glb_thr_alpha_max_noinf'
    # exp_name = 'glb_thr_hitrate_25max'
    # exp_name = 'glb_thr_alpha_25trace_zerogate' # TODO: is needed ???
    metric_per_task = OrderedDict({
        # 'mrpc': 'acc',
        'sst': 'acc',
        'wikitext': 'word_perplexity',
        # 'hellaswag': 'acc',
        # 'gsm8k': 'acc',
        # 'arc_easy': 'acc',
        # 'piqa': 'acc',
    })
    for task_name in metric_per_task:
        num_pruned = np.arange(1, 3)
        metrics = []
        log_dir = Path('results/moe') / model_name / task_name / exp_name
        log_dir.mkdir(exist_ok=True, parents=True)
        for num_experts_to_prune in num_pruned:
            is_prune = False if num_experts_to_prune == 0 else True
            ratio = num_experts_to_prune / total_num_experts
            try:
                print(f"Started experiment with {num_experts_to_prune} experts pruned on {task_name}\n")
                time_dict = {}
                start_time = time()
                results = evaluator.simple_evaluate(
                    model='hf-causal',
                    model_args=model_args,
                    tasks=[task_name],
                    num_fewshot=args.num_fewshot,
                    batch_size=args.batch_size,
                    max_batch_size=args.max_batch_size,
                    device=None,#args.device,
                    no_cache=args.no_cache,
                    limit=desc.limit,
                    description_dict=description_dict,
                    decontamination_ngrams_path=args.decontamination_ngrams_path,
                    check_integrity=args.check_integrity,
                    write_out=args.write_out,
                    output_base_path=args.output_base_path,
                    tokenizer=model_id,
                    is_prune=is_prune,
                    ratio=ratio,
                    exp_dir=log_dir
                )
                eval_time = time() - start_time
                time_dict['eval'] = eval_time
                print(f'eval took {eval_time} seconds')

                metric_name = metric_per_task[task_name]
                metric = results['results'][task_name][metric_name]
                if metric == 'acc':
                    metric *= 100
                metrics.append(metric)
                results['time'] = time_dict
                results['num_experts_to_prune'] = int(num_experts_to_prune)
                results['total_num_experts'] = total_num_experts
                results['experiment_config'] = desc.__dict__
                results_dir = log_dir if is_prune else log_dir.parent
                filename = f'results_r{ratio:.3f}.json'
                results_file = log_dir / filename
                print(results_file)
                all_results_paths.append(results_file.resolve())
            except Exception as error:
                print(traceback.print_exc())
                print(f"Eval of desc={desc} failed: {error}")
                continue
            finally:
                with results_file.open('w') as f:
                    json.dump(results, f, indent=2)
                print(evaluator.make_table(results))

        fp32_results = log_dir.parent / 'results_r0.000.json'
        if fp32_results.exists():
            shutil.copyfile(fp32_results, log_dir / 'results_r0.000.json')

        for path in all_results_paths:
            print(path, '\n')
            with path.open() as f:
                j = json.load(f)
                r = j['results']
                print(json.dumps(r, indent=4))

if __name__ == "__main__":
    main()

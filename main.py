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

from lm_eval import evaluator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
logging.getLogger("openai").setLevel(logging.WARNING)
from pathlib import Path
import threading
import numpy as np

LOGS_DIR = Path("./logs_compress")


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", required=True)
    # parser.add_argument(
    #     "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    # )
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=2)#100)
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

def plot_tradeoff(x, y, model_name, task_name, log_dir):
    # x=np.array([0,1,2,3,4,5])
    # y=np.array([70,71,72,73,74,75])
    fp32_ref = y[0]
    fig, ax = plt.subplots()
    ax.axhline(fp32_ref, color='red', linestyle='dotted')
    for xx,yy in zip(x,y):
        ax.text(xx, yy + .2, yy)
    ax.xaxis.grid()
    # ax.yaxis.grid()
    ax.plot(x,y, marker='o')
    plt.xlabel("Avg. number of pruned experts per layer")
    plt.ylabel("Accuracy")
    plt.title(f'Pruning trade-off for {model_name} on {task_name}')
    plt.savefig(log_dir / 'tradeoff.png')

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
    task_name = 'mrpc'
    # desc = ExpDesc("mistralai/Mixtral-8x7B-Instruct-v0.1", exp_name='fp16')
    desc = ExpDesc("dfurman/Mixtral-8x7B-Instruct-v0.1", exp_name='fp16')

    model_id = desc.model_id
    model_name = model_id.replace('/', '__')
    model_args = f'pretrained={model_id}'
    log_dir = Path('results/moe') / model_name / task_name
    total_num_experts = 8

    num_pruned = np.arange(0, 1)
    accuracies = []
    for num_experts_to_prune in num_pruned:
        is_prune = False if num_experts_to_prune == 0 else True
        ratio = num_experts_to_prune / total_num_experts
        try:
            print(f"Started experiment with ratio={ratio} is_prune={is_prune}\n")
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
                limit=2,#desc.limit,
                description_dict=description_dict,
                decontamination_ngrams_path=args.decontamination_ngrams_path,
                check_integrity=args.check_integrity,
                write_out=args.write_out,
                output_base_path=args.output_base_path,
                tokenizer=model_id,
                is_prune=is_prune,
                ratio=ratio
            )
            eval_time = time() - start_time
            time_dict['eval'] = eval_time
            print(f'eval took {eval_time} seconds')
            accuracies.append(results['results'][task_name]['acc'] * 100)
            results['time'] = time_dict
            results['num_experts_to_prune'] = int(num_experts_to_prune)
            results['total_num_experts'] = total_num_experts
            results['experiment_config'] = desc.__dict__
            results_file = log_dir / f'results_r{ratio:.3f}.json'
            print(results_file)
            all_results_paths.append(results_file.resolve())

        except Exception as error:
            print(traceback.print_exc())
            print(f"Eval of desc={desc} failed: {error}")
            continue
        finally:
            plot_tradeoff(x=num_pruned, y=np.array(accuracies), model_name=model_name, task_name=task_name, log_dir=log_dir)
            with results_file.open('w') as f:
                json.dump(results, f, indent=2)
            print(evaluator.make_table(results))

    for path in all_results_paths:
        print(path, '\n')
        with path.open() as f:
            j = json.load(f)
            r = j['results']
            print(json.dumps(r, indent=4))

if __name__ == "__main__":
    main()

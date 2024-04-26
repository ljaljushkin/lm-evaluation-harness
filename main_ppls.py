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
    # parser.add_argument("--model", default='stabilityai/stablelm-2-zephyr-1_6b')
    # parser.add_argument("--model", default='mistralai/Mistral-7B-v0.1')
    # parser.add_argument("--model", default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument("--model", default='stabilityai/stablelm-tuned-alpha-7b')

    # parser.add_argument(
    #     "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)

    # )
    parser.add_argument("--tuned_adapters_dir", type=str)
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=1)
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

    model_id = args.model
    model_args = f'pretrained={model_id}'

    tuned_adapters_dir = Path('/home/nlyaly/projects/lm-evaluation-harness/cache/stablelm-tuned-alpha-7b')
    DIRS = [
        '25.44_opt_search_loftq_ptb_synth_faster_R8_Ldd_VALIDATE',
        '25.58_opt_search_loftq_ptb_synth_R8_Ldd_VALIDATE',
        '25.98_opt_search_loftq_ptb_max_len_small_steps_R8_Ldd_VALIDATE',
    ]
    for exp_dir in DIRS:
        all_results_paths = []
        ppls = []
        adapters_dir = tuned_adapters_dir / exp_dir
        for idx in range(16):
            task_name = 'wikitext'
            adapter_dir = adapters_dir / str(idx)
            metrics = []
            # log_dir = Path('cache') / model_name / 'int4_via_nf4'
            # log_dir = Path('cache') / model_name / 'fp16'
            log_dir = adapter_dir
            log_dir.mkdir(exist_ok=True, parents=True)
            try:
                print(f"Started experiment on {task_name}\n")
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
                    limit=None,
                    description_dict=description_dict,
                    decontamination_ngrams_path=args.decontamination_ngrams_path,
                    check_integrity=args.check_integrity,
                    write_out=args.write_out,
                    output_base_path=args.output_base_path,
                    tokenizer=model_id,
                    tuned_adapters_dir=adapter_dir,
                )
                eval_time = time() - start_time
                time_dict['eval'] = eval_time
                print(f'eval took {eval_time} seconds')
                results['time'] = time_dict
                filename = f'results_{task_name}.json'
                results_file = log_dir / filename
                print(results_file)
                all_results_paths.append(results_file.resolve())
            except Exception as error:
                print(traceback.print_exc())
                continue
            finally:
                print(evaluator.make_table(results))
                with results_file.open('w') as f:
                    json.dump(results, f, indent=2)
                    word_ppl = results["results"][task_name]["word_perplexity"]
                ppls.append(word_ppl)

            for path in all_results_paths:
                print(path, '\n')
                with path.open() as f:
                    j = json.load(f)
                    r = j['results']
                    print(json.dumps(r, indent=4))
        print(ppls)
        plt.grid(axis='both', linestyle='-')
        xx = list(range(16))
        plt.xticks(xx)
        plt.plot(xx, ppls, **{'marker': 'o'}, label='')
        plt.savefig(adapters_dir / 'ppls_new.png')
        plt.clf()
if __name__ == "__main__":
    main()

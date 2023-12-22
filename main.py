import argparse
import gc
import json
import logging
import os
import sys

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

import torch
from transformers import AutoModelForCausalLM

from lm_eval import evaluator
from typing import Dict, Optional, Tuple, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.intel import OVModelForCausalLM

from optimum.intel.openvino import OVConfig, OVQuantizer

logging.getLogger("openai").setLevel(logging.WARNING)

import openvino.runtime as ov
from openvino import Core
import openvino
import queue
import atexit
from nncf import compress_weights
from pathlib import Path
import threading
import matplotlib.pyplot as plt
core = Core()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=100)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--device", type=str, default='cpu')
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
    is_bin_needed: bool = True

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

    use_pkv = True
    descs = [
    ]
    MODEL_IDS = [
        'facebook/opt-125m',
        # 'databricks/dolly-v2-3b',
        # 'openlm-research/open_llama_3b',
        # 'facebook/opt-6.7b',
        # 'bigscience/bloom-7b1',
        # 'bigscience/bloomz-7b1',
        # 'bigscience/bloomz-560m',
        # 'meta-llama/Llama-2-7b-chat-hf',
        # 'HuggingFaceH4/zephyr-7b-beta',
        # 'stable-zephyr-3b-dpo',
        # 'stabilityai/stablelm-3b-4e1t',
        # 'togethercomputer/RedPajama-INCITE-7B-Instruct',
        # 'meta-llama/Llama-2-13b-chat-hf',
        # 'databricks/dolly-v2-12b',
        # 'THUDM/chatglm2-6b'
        # 'THUDM/chatglm-6b',
        # 'Qwen/Qwen-7B-Chat',
    ]

    EXP_NAMES = [
        'int8',
        'int4_asym_g128_r80',
        'int4_asym_g128_r80_max_var',
    ]

    descs = [ExpDesc(model_id, exp_name=name) for model_id in MODEL_IDS for name in EXP_NAMES]
    print('All experiments summary:')
    for desc in descs:
        print(json.dumps(desc.__dict__,  indent=4))

    CACHE_DIR = Path('cache')

    all_results_paths = []
    for desc in descs:
        try:
            model_id = desc.model_id
            printable_desc = json.dumps(desc.__dict__,  indent=4)
            print(f"Started experiment {printable_desc}\n")
            model_name = Path(model_id).name.lower()
            random.seed(42)
            date = datetime.now().strftime("%b%d_%H-%M-%S")
            cache_dir = CACHE_DIR / model_name
            cache_dir.mkdir(parents=True, exist_ok=True)

            encoded_name = desc.get_encoded_name()
            model_args = f'pretrained={model_id}'

            log_dir = Path('runs') / model_name / f'{encoded_name}_{date}'
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / 'args.json').open('w') as f:
                f.write(printable_desc)

            ir_cache_dir = cache_dir / encoded_name
            ir_path = ir_cache_dir / 'openvino_model.bin'
            print(str(log_dir.resolve()))
            print(str(ir_path.resolve()))
            if desc.delete_ir_cache and ir_cache_dir.exists(): # ir_path.exists():
                # TODO: remove all except folder with results.json
                # shutil.rmtree(ir_cache_dir)
                print('remove IRs:')
                for file_to_remove in ir_cache_dir.glob('openvino_model.*'):
                    print(file_to_remove)
                    Path.unlink(file_to_remove)
            ir_cache_dir.mkdir(exist_ok=True)
            os.symlink(ir_cache_dir.resolve(), log_dir.resolve() / ir_cache_dir.name)
            os.symlink(log_dir.resolve(), ir_cache_dir.resolve() / log_dir.name)
            time_dict = {}

            print('ir path: ', ir_path)
            if not ir_path.exists():
                ov_model = OVModelForCausalLM.from_pretrained(model_id, use_cache=use_pkv, trust_remote_code=True)
                ov_model.save_pretrained(ir_cache_dir)
                del ov_model
                gc.collect()

            model_args = f'pretrained={ir_cache_dir.resolve()}'

            if desc.do_eval:
                start_time = time()
                results = evaluator.simple_evaluate(
                    model='optimum-causal',
                    model_args=model_args,
                    tasks=['lambada_openai'],
                    num_fewshot=args.num_fewshot,
                    batch_size=args.batch_size,
                    max_batch_size=args.max_batch_size,
                    device=args.device,
                    no_cache=args.no_cache,
                    limit=desc.limit,
                    description_dict=description_dict,
                    decontamination_ngrams_path=args.decontamination_ngrams_path,
                    check_integrity=args.check_integrity,
                    write_out=args.write_out,
                    output_base_path=args.output_base_path,
                    tokenizer=model_id,
                )
                eval_time = time() - start_time
                time_dict['eval'] = eval_time
                print(f'eval took {eval_time} seconds')
                results['time'] = time_dict
                results['experiment_config'] = desc.__dict__

                file_stats = ir_path.stat()
                file_size_gb = file_stats.st_size /  (1024 * 1024 * 1024)
                results['model_size'] = file_size_gb
                results['ov_version'] = str(openvino.__version__)
                results_file = log_dir / 'results.json'
                print(results_file)
                all_results_paths.append(results_file.resolve())

            model_cache_dir = ir_cache_dir / 'model_cache'
            if model_cache_dir.exists():
                shutil.rmtree(model_cache_dir)

            if not desc.is_bin_needed:
                Path.unlink(ir_path)
        except Exception as error:
            print(traceback.print_exc())
            print(f"Eval of desc={desc} failed: {error}")
            continue
        finally:
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

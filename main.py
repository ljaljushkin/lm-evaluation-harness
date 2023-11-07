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
# from visualization import parse_results

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from memory_profiler import memory_usage
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


import psutil


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
    parser.add_argument("--device", type=str, default='auto')
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

    # if args.tasks is None:
    #     task_names = tasks.ALL_TASKS
    # else:
    #     task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    # print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    use_pkv = True
    descs = [
        # ExpDesc('bigscience/bloom-7b1', exp_name='nf4_ov_g32_r80'),
        # ExpDesc('bigscience/bloom-7b1', exp_name='nf4_ov_g64_r60'),
        # ExpDesc('bigscience/bloom-7b1', exp_name='nf4_ov_g128_r60'),
        # ExpDesc('bigscience/bloom-7b1', exp_name='int4_ov_g32_r80'),
        # ExpDesc('bigscience/bloom-7b1', exp_name='int4_ov_g64_r60'),
        # ExpDesc('bigscience/bloom-7b1', exp_name='int4_ov_g128_r60'),
        # ExpDesc('togethercomputer/RedPajama-INCITE-7B-Instruct', exp_name='nf4_ov_g32_r80'),
        # ExpDesc('databricks/dolly-v2-3b', exp_name='int4_g128'),
        # ExpDesc('databricks/dolly-v2-3b', exp_name='nf4_ov_g128_r60'),
        # ExpDesc('meta-llama/Llama-2-13b-chat-hf', exp_name='nf4_ov_g128'),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', exp_name='int4_ov_g128_r80')

        # ExpDesc('facebook/opt-125m', exp_name='int4_g128'),
        # ExpDesc('facebook/opt-125m', exp_name='int4_g128_oc'),
        # ExpDesc('TheBloke/zephyr-7B-beta-GPTQ', is_fp32=True),
        ExpDesc('HuggingFaceH4/zephyr-7b-beta', is_fp32=True),
        # ExpDesc('facebook/opt-6.7b', exp_name='int4_ov_g64_r60'),
        # ExpDesc('facebook/opt-6.7b', exp_name='int4_ov_g32'),
        # ExpDesc('facebook/opt-6.7b', exp_name='int4_ov_g32_r80'),
        # ExpDesc('facebook/opt-6.7b', exp_name='int4_ov_g32_r60'),

        # ExpDesc('togethercomputer/RedPajama-INCITE-7B-Instruct', exp_name='int4_ov_g128'),
        # ExpDesc('togethercomputer/RedPajama-INCITE-7B-Instruct', exp_name='int4_ov_g128_r80'),
        # ExpDesc('databricks/dolly-v2-3b', exp_name='int4_ov_g64_r40'),
        # ExpDesc('databricks/dolly-v2-3b', exp_name='int4_ov_g32_r50'),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', exp_name='int4_ov_g128_nozp_r80'),
    ]
    MODEL_IDS = [
        # 'facebook/opt-125m',
        # 'databricks/dolly-v2-3b',
        # 'openlm-research/open_llama_3b',
        # 'facebook/opt-6.7b',
        # 'bigscience/bloom-7b1',
        # 'meta-llama/Llama-2-7b-chat-hf',
        # 'togethercomputer/RedPajama-INCITE-7B-Instruct',
        # 'meta-llama/Llama-2-13b-chat-hf',
        # 'databricks/dolly-v2-12b',
        # 'THUDM/chatglm2-6b'
        # 'THUDM/chatglm-6b'
    ]

    EXP_NAMES = [
        # 'nf4_ov_g128',
        # 'int4_ov_g128_data',
        # 'int4_ov_g128',
        # "int4_ov_g64_nozp",
        # "int4_ov_g64_nozp_data",
        # "int4_ov_g64_nozp_r80",
        # "int4_ov_g64_nozp_r80_data",
        'int8',
        # 'fp32',
        # 'int4_g128',
        # 'int4_g128_nozp',
        # 'int4_g128_nozp_r80',
    ]

    # descs = [Ex4pDesc(model_id, exp_name=name) for model_id in MODEL_IDS for name in EXP_NAMES]

    all_results_paths = []
    for desc in descs:
        try:
            model_id = desc.model_id
            printable_desc = json.dumps(desc.__dict__,  indent=4)
            print(f"Started experiment {printable_desc}\n")
            model_name = Path(model_id).name
            random.seed(42)
            date = datetime.now().strftime("%b%d_%H-%M-%S")
            cache_dir = Path('cache') / model_name
            cache_dir.mkdir(parents=True, exist_ok=True)

            encoded_name = desc.get_encoded_name()
            model_args = f'pretrained={model_id}'

            log_dir = Path('runs') / model_name / f'{encoded_name}_{date}'
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / 'args.json').open('w') as f:
                f.write(printable_desc)

            # ir_cache_dir = cache_dir / encoded_name
            # ir_path = ir_cache_dir / 'openvino_model.bin'
            # print(str(log_dir.resolve()))
            # print(str(ir_path.resolve()))
            # if desc.delete_ir_cache and ir_cache_dir.exists(): # ir_path.exists():
            #     # TODO: remove all except folder with results.json
            #     # shutil.rmtree(ir_cache_dir)
            #     print('remove IRs:')
            #     for file_to_remove in ir_cache_dir.glob('openvino_model.*'):
            #         print(file_to_remove)
            #         Path.unlink(file_to_remove)
            # ir_cache_dir.mkdir(exist_ok=True)
            # os.symlink(ir_cache_dir.resolve(), log_dir.resolve() / ir_cache_dir.name)
            # os.symlink(log_dir.resolve(), ir_cache_dir.resolve() / log_dir.name)
            time_dict = {}

            # if not ir_path.exists():
            #     if 'fp32' not in encoded_name:
            #         print(f'started weights compression')
            #         start_time = time()
            #         quantization_config = {
            #             "algorithm": "quantization"
            #         }
            #         model = AutoModelForCausalLM.from_pretrained(
            #             model_id, use_cache=use_pkv, trust_remote_code=True,
            #             # TODO: aidova tip to avoid issue with model.onnx and probably with compilation
            #             # torchscript=True,
            #             use_auth_token=True
            #         )
            #         print(model)
            #         tokenizer = AutoTokenizer.from_pretrained(model_id)

            #         config = OVConfig(compression=quantization_config)
            #         config.target_device = "TRIAL"
            #         tokenizer.pad_token = tokenizer.eos_token

            #         quantizer = OVQuantizer.from_pretrained(model)

            #         if hasattr(model, "transformer") and hasattr(model.transformer, "wte") and type(model.transformer.wte) != torch.nn.Embedding:
            #             from nncf.torch import register_module
            #             register_module(ignored_algorithms=[], target_weight_dim_for_compression=1)(type(model.transformer.wte))

            #         start_memory_logging_routine(log_dir)
            #         quantizer.quantize(
            #             save_directory=ir_cache_dir, weights_only=True,
            #             group_size=desc.group_size, mode=desc.mode, is_mixed=desc.is_mixed
            #         )

            #         nncf_time = time() - start_time
            #         time_dict['nncf'] = nncf_time
            #         print(f'weights compression took {nncf_time} seconds')
            #         del model
            #     else:
            #         ov_model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=use_pkv, trust_remote_code=True, from_transformers=True)
            #         ov_model.save_pretrained(ir_cache_dir)
            #         del ov_model
            #     gc.collect()

            # model_args = f'pretrained={ir_cache_dir.resolve()}'

            if desc.do_eval:
                start_time = time()
                results = evaluator.simple_evaluate(
                    model='hf-causal',
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
                    tokenizer=model_id
                )
                eval_time = time() - start_time
                time_dict['eval'] = eval_time
                print(f'eval took {eval_time} seconds')
                results['time'] = time_dict
                results['experiment_config'] = desc.__dict__

                # file_stats = ir_path.stat()
                # file_size_gb = file_stats.st_size /  (1024 * 1024 * 1024)
                # results['model_size'] = file_size_gb
                # results['ov_version'] = str(openvino.__version__)
                results_file = log_dir / 'results.json'
                print(results_file)
                all_results_paths.append(results_file.resolve())

            model_cache_dir = ir_cache_dir / 'model_cache'
            if model_cache_dir.exists():
                shutil.rmtree(model_cache_dir)

            # if not desc.is_bin_needed:
            #     Path.unlink(ir_path)
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

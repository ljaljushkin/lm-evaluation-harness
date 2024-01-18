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
from transformers import AutoConfig
# from optimum.intel.openvino import OVChatGLM2Model

from lm_eval import evaluator
# from visualization import parse_results
from typing import Dict, Optional, Tuple, List
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
    parser.add_argument("--device", type=str, default='gpu')
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

from optimum.utils import (
    NormalizedTextConfig, NormalizedConfigManager
)

# TasksManager._SUPPORTED_MODEL_TYPE["mistral"] = TasksManager._SUPPORTED_MODEL_TYPE['llama']

NormalizedConfigManager._conf['mistral'] = NormalizedTextConfig.with_args(num_key_value_heads='num_key_value_heads', allow_new=True)

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
        # ExpDesc('databricks/dolly-v2-3b', exp_name='nf4_ov_g128_r80'),
        # ExpDesc('databricks/dolly-v2-3b', exp_name='nf4_ov_g128_r60'),
        # ExpDesc('meta-llama/Llama-2-13b-chat-hf', exp_name='nf4_ov_g128'),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', exp_name='int4_ov_g128_r80')

        # ExpDesc('facebook/opt-6.7b', exp_name='int4_g128'),
        #ExpDesc('facebook/opt-6.7b', exp_name='int4_ov_g64_r80'),
        #ExpDesc('facebook/opt-6.7b', exp_name='int4_ov_g64_r60'),
        #ExpDesc('facebook/opt-6.7b', exp_name='int4_ov_g32'),
        #ExpDesc('facebook/opt-6.7b', exp_name='int4_ov_g32_r80'),
        #ExpDesc('facebook/opt-6.7b', exp_name='int4_ov_g32_r60'),

        # ExpDesc('togethercomputer/RedPajama-INCITE-7B-Instruct', exp_name='int4_ov_g128'),
        # ExpDesc('togethercomputer/RedPajama-INCITE-7B-Instruct', exp_name='int4_ov_g128_r80'),
        # ExpDesc('databricks/dolly-v2-3b', exp_name='int4_ov_g64_r40'),
        # ExpDesc('databricks/dolly-v2-3b', exp_name='int4_ov_g32_r50'),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', exp_name='int4_ov_g128_nozp_r80'),
        # ExpDesc('HuggingFaceH4/zephyr-7b-beta', exp_name='int4_g128_r80'),
        # ExpDesc('bigscience/bloomz-7b1', exp_name='int4_g128_r80'),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', exp_name='int4_g128_nozp_r80'),

        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', exp_name='fp16'),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', exp_name='int4_g128_nozp_r80'),
        # ExpDesc('HuggingFaceH4/zephyr-7b-beta', exp_name='fp32'),
        # ExpDesc('HuggingFaceH4/zephyr-7b-beta', exp_name='int4_g128_r80'),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', exp_name='int4_g128_nozp_r80_criteria_IN1'),
        # ExpDesc('HuggingFaceH4/zephyr-7b-beta', exp_name='int4_g128_r80_criteria_IN1'),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', exp_name='int4_g128_nozp_r80_criteria_OUT2'),
        # ExpDesc('HuggingFaceH4/zephyr-7b-beta', exp_name='int4_g128_r80_criteria_OUT2'),
        # ExpDesc('bigscience/bloomz-7b1', exp_name='int4_g128_r80_criteria_OUT2'),
        # ExpDesc('stable-zephyr-3b-dpo', exp_name='int8'),

        # ExpDesc('stable-zephyr-3b-dpo', exp_name='int4_g64_nozp_r80_criteria_OUT2'),
        # ExpDesc('stable-zephyr-3b-dpo', exp_name='int4_g64_r80_criteria_OUT2'),
        # ExpDesc('stable-zephyr-3b-dpo', exp_name='int4_g64_nozp_r80'),
        # ExpDesc('stable-zephyr-3b-dpo', exp_name='int4_g64_r80'),
        # ExpDesc('stable-zephyr-3b-dpo', exp_name='int4_g64_nozp_r80_criteria_IN'),
        # ExpDesc('stable-zephyr-3b-dpo', exp_name='int4_g64_r80_criteria_IN'),

        # ExpDesc('databricks/dolly-v2-3b', exp_name='int4_asym_g32_r50_max_var'),
        # ExpDesc('databricks/dolly-v2-3b', exp_name='int4_asym_g32_r50'),
        # ExpDesc('facebook/opt-6.7b', exp_name='int4_asym_g64_r80_max_var'),
        # ExpDesc('facebook/opt-6.7b', exp_name='int4_asym_g64_r80'),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', exp_name='int4_asym_g128_r80_max_var'),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', exp_name='int4_asym_g128_r80'),
        # ExpDesc('meta-llama/Llama-2-13b-chat-hf', exp_name='int4_sym_g64_r80_max_var'),
        # ExpDesc('meta-llama/Llama-2-13b-chat-hf', exp_name='int4_sym_g64_r80'),
    ]
    MODEL_IDS = [
        # 'facebook/opt-125m',
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
        # 'THUDM/chatglm-6b',
        # 'THUDM/chatglm2-6b',
        'THUDM/chatglm3-6b',
        # 'Qwen/Qwen-7B-Chat',
    ]

    EXP_NAMES = [
        # 'gptq',
        'fp16',
        # 'int8',
        # 'int4_asym_g128_r80',
        # 'int4_asym_g128_r80_max_var',
        # 'int4_g64_r60_hawq_in',
        # 'int4_g64_nozp_r60_hawq_in',
        # 'int4_g64_r80_hawq_in',
        # 'int4_g64_nozp_r80_hawq_in',

        # 'int4_g128_r60_hawq_in',
        # 'int4_g128_nozp_r60_hawq_in',
        # 'int4_sym_g128_r80',
        # 'int4_sym_g128_r80_max_var',
        # 'int4_g64_nozp_r80_greedy1',
        # 'int4_g64_nozp_r80_mean_var',
        # 'int4_g64_nozp_r80_max_var',
        # 'int4_g64_nozp_r80_mean_max',
        # 'int4_sym_g128_r80',
        # 'int4_sym_g128_r80_hawq_in',
        # 'int4_sym_g128_r80_max_var',

        # 'int4_sym_g64_r80_pr',
        # 'int4_sym_g64_r80_max_activation_variance'
        # 'int4_sym_g64_r80_max_var',
        # 'int4_sym_g64_r80_weight_quantization_error',

        'int4_sym_g128',
        'int4_sym_g128_r80',
        'int4_sym_g128_r60',
        'int4_sym_g128_r40',
        'int4_sym_g128_r20',
        'int8_sym',

        # 'int4_sym_g128_r80_mean_var',
        # 'int4_sym_g128_r80_mean_max',
        # 'int4_g128_nozp_r80',
        # 'int4_g128_nozp',
        # 'int4_g128',
        # 'int4_g128_r80',
        # 'int4_g64_nozp',
        # 'int4_g128',
        # 'int4_g128_r80',
        # 'int4_g128_r60',
        # 'int4_g64',
        # 'int4_g64_r80',
        # 'int4_g64_r60',
        # 'int4_g128_nozp',
        # 'int4_g128_nozp_r60',
        # 'int4_g64_nozp_r80',
        # 'int4_g64_nozp_r60',
        # 'int4_g128_r80',
        # 'int4_g128_r80_criteria'
    ]

    descs = [ExpDesc(model_id, exp_name=name) for model_id in MODEL_IDS for name in EXP_NAMES]

    from transformers.generation import GenerationConfig
    from optimum.utils import (
       NormalizedTextConfig, NormalizedConfigManager
    )
    from optimum.exporters import TasksManager
    NormalizedConfigManager._conf['qwen'] = NormalizedTextConfig.with_args(
        num_layers='num_hidden_layers', num_attention_heads='num_attention_heads', hidden_size='hidden_size')
    NormalizedConfigManager._conf['chatglm'] = NormalizedTextConfig.with_args(num_layers='num_layers', num_attention_heads='num_attention_heads')
    TasksManager._SUPPORTED_MODEL_TYPE["stablelm-epoch"] = TasksManager._SUPPORTED_MODEL_TYPE["llama"]
    stable_lm_config = NormalizedTextConfig.with_args(
        num_layers='num_hidden_layers',
        num_attention_heads='num_attention_heads'
    )
    NormalizedConfigManager._conf['stablelm_epoch'] = stable_lm_config
    NormalizedConfigManager._conf["stablelm-epoch"] = stable_lm_config

    print('All experiments summary:')
    for desc in descs:
        print(json.dumps(desc.__dict__,  indent=4))

    CACHE_DIR = Path(os.readlink('cache'))

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
                if 'fp16' not in encoded_name:
                    print(f'started weights compression')
                    start_time = time()
                    quantization_config = {
                        "algorithm": "quantization"
                    }
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, use_cache=use_pkv, trust_remote_code=True,
                        # TODO: aidova tip to avoid issue with model.onnx and probably with compilation
                        # torchscript=True,
                        use_auth_token=True
                    )
                    print(model)
                    tokenizer = AutoTokenizer.from_pretrained(model_id)

                    config = OVConfig(compression=quantization_config)
                    config.target_device = "TRIAL"
                    tokenizer.pad_token = tokenizer.eos_token

                    quantizer = OVQuantizer.from_pretrained(model)

                    if hasattr(model, "transformer") and hasattr(model.transformer, "wte") and type(model.transformer.wte) != torch.nn.Embedding:
                        from nncf.torch import register_module
                        register_module(ignored_algorithms=[], target_weight_dim_for_compression=1)(type(model.transformer.wte))

                    quantizer.quantize(
                        save_directory=ir_cache_dir, weights_only=True,
                        group_size=desc.group_size, mode=desc.mode, is_mixed=desc.is_mixed
                    )

                    nncf_time = time() - start_time
                    time_dict['nncf'] = nncf_time
                    print(f'weights compression took {nncf_time} seconds')
                    del model
                else:
                    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                    ov_model = OVModelForCausalLM.from_pretrained(model_id, config=config, use_cache=use_pkv, trust_remote_code=True, export=True)
                    # ov_model = OVModelForCausalLM.from_pretrained(model_id, use_cache=use_pkv, trust_remote_code=True, export=True)
                    # ov_model = OVModelForCausalLM.from_pretrained(model_id, use_cache=use_pkv, trust_remote_code=True)
                    # config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                    # ov_model = OVChatGLM2Model.from_pretrained(
                    #     model_id,
                    #     config=config,
                    #     # revision=revision,
                    #     trust_remote_code=True,
                    #     use_cache=True,
                    #     # from_transformers=True
                    # )
                    ov_model.save_pretrained(ir_cache_dir)
                    del ov_model
                gc.collect()

            model_args = f'pretrained={ir_cache_dir.resolve()}'

            if desc.do_eval:
                start_time = time()
                results = evaluator.simple_evaluate(
                    model='optimum-causal',
                    # model='hf-causal',
                    model_args=model_args,
                    # tasks=['lambada_openai'],
                    # tasks=['triviaqa'],
                    # tasks=['wikitext'],
                    tasks=['wikitext_zh_yue_clean_no_small'],
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
                    # tokenizer=ir_cache_dir.resolve()
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

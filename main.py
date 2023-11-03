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


import psutil

memory_data_queue = queue.Queue()
monitoring_thread_should_stop = False

LOGS_DIR = Path("./logs_compress")


def stop_monitoring_thread():
    global monitoring_thread_should_stop
    monitoring_thread_should_stop = True


def monitor_memory(q):
    while not monitoring_thread_should_stop:
        memory_usage = psutil.Process().memory_info().rss >> 20     # MB
        timestamp = datetime.now()
        (datetime.now() - timestamp).total_seconds()
        q.put((timestamp, memory_usage))
        sleep(1)


def log_memory_usage(log_dir):
    memory_usage_data = []
    while not memory_data_queue.empty():
        timestamp, memory_usage = memory_data_queue.get()
        memory_usage_data.append((timestamp, memory_usage))

    # Save the memory usage data to a file
    with open(log_dir / 'memory_usage_log.txt', 'w') as log_file:
        for timestamp, memory_usage in memory_usage_data:
            log_file.write(f"{timestamp} {memory_usage}\n")

        log_file.writelines([
            f"Total time: {(memory_usage_data[-1][0] - memory_usage_data[0][0]).total_seconds() // 60} (minutes)\n",
            f"Max memory: {max(tuple(zip(*memory_usage_data))[1])} (MB)"])

    timestamps, memory_usage = zip(*memory_usage_data)
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, memory_usage)
    plt.xlabel("Time")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage vs. Time")
    plt.grid(True)
    plt.savefig(log_dir / "memory_usage.png")

def start_memory_logging_routine(log_dir):
    memory_monitor_thread = threading.Thread(target=monitor_memory, args=(memory_data_queue,))
    memory_monitor_thread.daemon = True  # Daemonize the thread
    memory_monitor_thread.start()
    atexit.register(lambda: [stop_monitoring_thread(), memory_monitor_thread.join(), log_memory_usage(log_dir)])

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
    limit: float = 100
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

from optimum.exporters import TasksManager
from optimum.exporters.tasks import make_backend_config_constructor_for_task
from optimum.exporters.onnx.config import TextDecoderOnnxConfig
from optimum.utils import (
    NormalizedTextConfig, NormalizedConfigManager, DEFAULT_DUMMY_SHAPES,
    DummyPastKeyValuesGenerator,
    DummyTextInputGenerator,
)

class TextDecoderWithPositionIdsOnnxConfig(TextDecoderOnnxConfig):
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs

        # Decoders based on GPT2 require a position_ids input to avoid
        # generating wrong position_ids in the model itself:
        # https://github.com/huggingface/transformers/blob/v4.33.1/src/transformers/models/gpt2/modeling_gpt2.py#L802
        if not self.no_position_ids and "text-generation" in self.task:
            common_inputs["position_ids"] = {0: "batch_size", 1: "sequence_length"}

        return common_inputs

class MistralDummyTextInputGenerator(DummyTextInputGenerator):
    SUPPORTED_INPUT_NAMES = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
    }

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        input = super().generate(input_name, framework, int_dtype, float_dtype)
        if input_name == "position_ids":
            input = input[:, -1:]
        return input

class MistralDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
        )
        self.num_key_value_heads = normalized_config.num_key_value_heads

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class MistralOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    # The ONNX export of this architecture needs the Trilu operator support, available since opset 14
    DEFAULT_ONNX_OPSET = 14
    DUMMY_INPUT_GENERATOR_CLASSES = (
        MistralDummyTextInputGenerator,
        MistralDummyPastKeyValuesGenerator,
    )
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_key_value_heads="num_key_value_heads", allow_new=True)
    no_position_ids = False

# TasksManager._SUPPORTED_MODEL_TYPE["mistral"] = TasksManager._SUPPORTED_MODEL_TYPE['llama']

export_config = MistralOnnxConfig
TasksManager._SUPPORTED_MODEL_TYPE['mistral'] = {
    'onnx': {
        'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
        'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
    },
    'openvino': {
        'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
        'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
    },
}

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

        ExpDesc('HuggingFaceH4/zephyr-7b-beta', exp_name='int4_g128'),
        ExpDesc('HuggingFaceH4/zephyr-7b-beta', exp_name='fp32'),
        ExpDesc('HuggingFaceH4/zephyr-7b-beta', exp_name='int4_g128_nozp'),
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

            if not ir_path.exists():
                if 'fp32' not in encoded_name:
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

                    start_memory_logging_routine(log_dir)
                    quantizer.quantize(
                        save_directory=ir_cache_dir, weights_only=True,
                        group_size=desc.group_size, mode=desc.mode, is_mixed=desc.is_mixed
                    )

                    nncf_time = time() - start_time
                    time_dict['nncf'] = nncf_time
                    print(f'weights compression took {nncf_time} seconds')
                    del model
                else:
                    ov_model = OVModelForCausalLM.from_pretrained(model_id, use_cache=use_pkv, trust_remote_code=True, from_transformers=True)
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
                    tokenizer=model_id
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
                with results_file.open('w') as f:
                    json.dump(results, f, indent=2)
                print(evaluator.make_table(results))

            model_cache_dir = ir_cache_dir / 'model_cache'
            if model_cache_dir.exists():
                shutil.rmtree(model_cache_dir)

            if not desc.is_bin_needed:
                Path.unlink(ir_path)
        except Exception as error:
            print(traceback.print_exc())
            print(f"Eval of desc={desc} failed: {error}")
            continue

    for path in all_results_paths:
        print(path, '\n')
        with path.open() as f:
            j = json.load(f)
            r = j['results']
            print(json.dumps(r, indent=4))

if __name__ == "__main__":
    main()

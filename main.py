import argparse
import json
import logging
import os
import sys

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
    limit: float = None
    is_mixed: bool = False
    do_eval: bool = True
    delete_ir_cache: bool = False
    is_fp32: bool = False
    exp_name: str = None

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
        # NOTE: CLX
        # ExpDesc('databricks/dolly-v2-3b', group_size=64, mode='nf4'),
        # ExpDesc('databricks/dolly-v2-3b', group_size=64, mode='uni'),
        # ExpDesc('databricks/dolly-v2-3b', group_size=64, mode='pq' ),
        # ExpDesc('openlm-research/open_llama_3b', group_size=64, mode='nf4', delete_ir_cache=True),
        # ExpDesc('openlm-research/open_llama_3b', group_size=64, mode='uni', delete_ir_cache=True),
        # ExpDesc('openlm-research/open_llama_3b', group_size=64, mode='pq', delete_ir_cache=True),

        # # parallel: 1
        # ExpDesc('facebook/opt-6.7b', is_fp32=True),
        # # parallel: 2
        # ExpDesc('facebook/opt-6.7b', group_size=64, mode='nf4'),
        # ExpDesc('facebook/opt-6.7b', group_size=64, mode='uni'),
        # ExpDesc('facebook/opt-6.7b', group_size=64, mode='pq'),
        # ExpDesc('facebook/opt-6.7b', group_size=128, mode='nf4'),
        # ExpDesc('facebook/opt-6.7b', group_size=128, mode='uni'),
        # ExpDesc('facebook/opt-6.7b', group_size=128, mode='pq'),

        # # ExpDesc('facebook/opt-6.7b', group_size=-1, mode='nf4'),
        # ExpDesc('facebook/opt-6.7b', group_size=-1, mode='uni'),
        # ExpDesc('facebook/opt-6.7b', group_size=-1, mode='pq'),

        # NOTE: SPR
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', group_size=64, mode='nf4', delete_ir_cache=True),
        # ExpDesc('meta-llama/Llama-2-13b-chat-hf', group_size=64, mode='nf4'),
        # ExpDesc('togethercomputer/RedPajama-INCITE-7B-Instruct', group_size=64, mode='nf4'),
        # ExpDesc('meta-llama/Llama-2-13b-chat-hf', group_size=64, mode='uni'),
        # ExpDesc('meta-llama/Llama-2-13b-chat-hf', group_size=64, mode='pq'),
        # ExpDesc('togethercomputer/RedPajama-INCITE-7B-Instruct', group_size=64, mode='uni', delete_ir_cache=True),
        # ExpDesc('togethercomputer/RedPajama-INCITE-7B-Instruct', group_size=64, mode='pq', delete_ir_cache=True),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', group_size=64, mode='uni',delete_ir_cache=True),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', group_size=64, mode='pq',delete_ir_cache=True),

        # parallel: 1
        # ExpDesc('databricks/dolly-v2-12b', is_fp32=True, do_eval=False),

        # parallel: 2
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', group_size=128, mode='nf4'),
        # ExpDesc('meta-llama/Llama-2-13b-chat-hf', group_size=128, mode='nf4'),
        # ExpDesc('meta-llama/Llama-2-13b-chat-hf', group_size=128, mode='uni'),
        # ExpDesc('meta-llama/Llama-2-13b-chat-hf', group_size=128, mode='pq'),
        # ExpDesc('databricks/dolly-v2-12b', is_fp32=True, do_eval=True),
        # ExpDesc('databricks/dolly-v2-12b', group_size=128, mode='nf4'),
        # ExpDesc('databricks/dolly-v2-12b', group_size=128, mode='uni'),
        # ExpDesc('databricks/dolly-v2-12b', group_size=128, mode='pq' ),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', group_size=128, mode='uni'),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', group_size=128, mode='pq'),

        # ExpDesc('databricks/dolly-v2-12b', group_size=64, mode='nf4'),
        # ExpDesc('databricks/dolly-v2-12b', group_size=64, mode='uni'),
        # ExpDesc('databricks/dolly-v2-12b', group_size=64, mode='pq' ),

        # ExpDesc('bigscience/bloom-7b1', is_fp32=True),
        # ExpDesc('bigscience/bloom-7b1', group_size=64, mode='nf4'),
        # ExpDesc('bigscience/bloom-7b1', group_size=64, mode='uni'),
        # ExpDesc('bigscience/bloom-7b1', group_size=64, mode='pq'),
        # ExpDesc('bigscience/bloom-7b1', group_size=128, mode='nf4'),
        # ExpDesc('bigscience/bloom-7b1', group_size=128, mode='uni'),
        # ExpDesc('bigscience/bloom-7b1', group_size=128, mode='pq'),

        # ExpDesc('databricks/dolly-v2-12b', is_fp32=True),
        # ExpDesc('meta-llama/Llama-2-13b-chat-hf', group_size=-1, mode='uni'),
        # ExpDesc('meta-llama/Llama-2-13b-chat-hf', group_size=-1, mode='nf4'),
        # ExpDesc('meta-llama/Llama-2-13b-chat-hf', group_size=-1, mode='pq'),
        # ExpDesc('bigscience/bloom-7b1', group_size=-1, mode='nf4'),
        # ExpDesc('bigscience/bloom-7b1', group_size=-1, mode='pq'),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', group_size=-1, mode='nf4', delete_ir_cache=True),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', group_size=-1, mode='pq'),
        # ExpDesc('meta-llama/Llama-2-7b-chat-hf', is_fp32=True, do_eval=False),
        # ExpDesc('databricks/dolly-v2-3b', is_fp32=True, do_eval=False),

        # ExpDesc('facebook/opt-125m', is_fp32=True),
        # ExpDesc('databricks/dolly-v2-3b', do_eval=True, delete_ir_cache=False, exp_name='int8', limit=100),
        # ExpDesc('meta-llama/Llama-2-13b-chat-hf', do_eval=True, delete_ir_cache=False, exp_name='nf4_ov'),
        ExpDesc('meta-llama/Llama-2-13b-chat-hf', do_eval=True, delete_ir_cache=True, group_size=-1, mode='nf4'),
        # ExpDesc('facebook/opt-125m', do_eval=True, delete_ir_cache=False, exp_name='nf4_ov_g128', limit=100),
    ]
    all_results_paths = []
    for desc in descs:
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
        log_file_path = log_dir / "log.txt"
        log_file = open(log_file_path, "w")
        try:
            sys.stdout = log_file
            sys.stderr = log_file

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
                else:
                    ov_model = OVModelForCausalLM.from_pretrained(model_id, use_cache=use_pkv, trust_remote_code=True, from_transformers=True)
                    ov_model.save_pretrained(ir_cache_dir)
        finally:
            log_file.flush()
            log_file.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

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
            results_file = log_dir / 'results.json'
            print(results_file)
            all_results_paths.append(results_file.resolve())
            with results_file.open('w') as f:
                json.dump(results, f, indent=2)
            print(evaluator.make_table(results))

        model_cache_dir = ir_cache_dir / 'model_cache'
        if model_cache_dir.exists():
            shutil.rmtree(model_cache_dir)

    for path in all_results_paths:
        print(path, '\n')
        with path.open() as f:
            j = json.load(f)
            r = j['results']
            print(json.dumps(r, indent=4))

if __name__ == "__main__":
    main()

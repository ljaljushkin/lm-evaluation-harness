from dataclasses import dataclass
from functools import partial
import gc
import shutil
from typing import Callable
import openvino.runtime as ov
from openvino import Core
import time
import queue
import atexit
import datetime
from nncf import compress_weights
from pathlib import Path
import threading
import matplotlib.pyplot as plt
from nncf.parameters import CompressWeightsMode
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
        timestamp = datetime.datetime.now()
        (datetime.datetime.now() - timestamp).total_seconds()
        q.put((timestamp, memory_usage))
        time.sleep(1)


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


# from optimum.intel import OVModelForCausalLM
# MODEL_NAME = 'opt-125m'
# use_pkv = True
# ov_model = OVModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', use_cache=use_pkv, trust_remote_code=True, from_transformers=True)
# ov_model.save_pretrained('/home/nlyaly/projects/nncf/tests/openvino')
# ie = ov.Core()

@dataclass
class ExpDesc:
    model: str
    compress_fn: Callable
    exp_name: str
    is_bin_needed: bool = False

int8_fn = compress_weights
nf4_fn = partial(compress_weights, mode=CompressWeightsMode.COMPRESSED_NF4, ratio=1, group_size=-1)
nf4_g128_fn = partial(compress_weights, mode=CompressWeightsMode.COMPRESSED_NF4, ratio=1, group_size=128)
mixed_g128_fn = partial(compress_weights, mode=CompressWeightsMode.COMPRESSED_NF4, ratio=0.5, group_size=128)

cache_dir = Path('cache')
model_name = 'openvino_model.xml'
EXP_DESCS = [
    # ExpDesc('Llama-2-13b-chat-hf', mixed_g128_fn, 'mixed_ov_g128'),
    # ExpDesc('Llama-2-7b-chat-hf', mixed_g128_fn, 'mixed_ov_g128'),
    # ExpDesc('Llama-2-13b-chat-hf', nf4_g128_fn, 'nf4_ov_g128', is_bin_needed=True),
    # ExpDesc('Llama-2-7b-chat-hf', nf4_g128_fn, 'nf4_ov_g128', is_bin_needed=True),
    # ExpDesc('Llama-2-13b-chat-hf', nf4_fn, 'nf4_ov', is_bin_needed=True),
    # ExpDesc('Llama-2-7b-chat-hf', nf4_fn, 'nf4_ov', is_bin_needed=True),
    ExpDesc('bloom-7b1', nf4_g128_fn, 'nf4_ov_g128', is_bin_needed=True),
    ExpDesc('opt-6.7b', nf4_g128_fn, 'nf4_ov_g128', is_bin_needed=True),
    ExpDesc('RedPajama-INCITE-7B-Instruct', nf4_g128_fn, 'nf4_ov_g128', is_bin_needed=True),
    ExpDesc('dolly-v2-12b', nf4_g128_fn, 'nf4_ov_g128', is_bin_needed=True),
    ExpDesc('dolly-v2-12b', nf4_fn, 'nf4_ov', is_bin_needed=True),
    # CLX
    # ExpDesc('dolly-v2-3b', nf4_g128_fn, 'nf4_ov_g128', is_bin_needed=True),
    # ExpDesc('dolly-v2-3b', nf4_fn, 'nf4_ov', is_bin_needed=True),
    # ExpDesc('open_llama_3b', nf4_g128_fn, 'nf4_ov_g128', is_bin_needed=True),
    # ExpDesc('open_llama_3b', nf4_fn, 'nf4_ov', is_bin_needed=True),
    # 'opt-125m's
]

start_memory_logging_routine(Path('./'))
for desc in EXP_DESCS:
    SRC_PATH = cache_dir / desc.model / 'fp32'/  model_name
    DST_PATH = cache_dir / desc.model / desc.exp_name /  model_name
    DST_PATH.parent.mkdir(exist_ok=True)
    shutil.copyfile(SRC_PATH.parent / 'config.json', DST_PATH.parent / 'config.json')

    model = core.read_model(model=SRC_PATH)
    start = time.time()

    model = desc.compress_fn(model)
    print(f'compressing weights took {(time.time() - start):.1f} seconds')
    start = time.time()
    ov.save_model(model, DST_PATH, compress_to_fp16=False)
    print(f"saving model {DST_PATH} took {(time.time() - start):.1f} seconds")
    if not desc.is_bin_needed:
        file_to_remove = DST_PATH.rename(DST_PATH.with_suffix('.bin'))
        Path.unlink(file_to_remove)

    del model
    gc.collect()

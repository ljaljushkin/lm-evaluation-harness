import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from nncf.common.utils.os import is_windows
from nncf import compress_weights, CompressWeightsMode, IgnoredScope
import openvino as ov
from pathlib import Path

class Command:
    def __init__(self, cmd: str, cwd: Path = None, env: Dict = None):
        self.cmd = cmd
        self.process = None
        self.exec_time = -1
        self.output = []  # store output here
        self.kwargs = {}
        self.timeout = False
        self.cwd = cwd
        self.env = env if env is not None else os.environ.copy()
        self.thread_exc = None

        # set system/version dependent "start_new_session" analogs
        if is_windows():
            self.kwargs.update(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        if sys.version_info < (3, 2):  # assume posix
            self.kwargs.update(preexec_fn=os.setsid)
        else:  # Python 3.2+ and Unix
            self.kwargs.update(start_new_session=True)

    def kill_process_tree(self, pid):
        try:
            if is_windows():
                os.killpg(pid, signal.SIGKILL)
            else:
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)])
        except OSError as err:
            print(err)

    def run(self, timeout=3600, assert_returncode_zero=True, stdout=True):
        print(f"Running command: {self.cmd}")

        def target():
            try:
                start_time = time.time()
                with subprocess.Popen(
                    self.cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    bufsize=1,
                    cwd=self.cwd,
                    env=self.env,
                    **self.kwargs,
                ) as p:
                    self.process = p
                    self.timeout = False

                    self.output = []
                    for line in self.process.stdout:
                        line = line.decode("utf-8")
                        self.output.append(line)
                        if stdout:
                            sys.stdout.write(line)

                    if stdout:
                        sys.stdout.flush()
                    self.process.stdout.close()

                    self.process.wait()
                    self.exec_time = time.time() - start_time
            except Exception as e:
                self.thread_exc = e

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)

        if self.thread_exc is not None:
            raise self.thread_exc

        if thread.is_alive():
            try:
                print("Error: process taking too long to complete--terminating" + ", [ " + self.cmd + " ]")
                self.kill_process_tree(self.process.pid)
                self.exec_time = timeout
                self.timeout = True
                thread.join()
            except OSError as e:
                print(self.process.pid, "Exception when try to kill task by PID, " + e.strerror)
                raise
        returncode = self.process.wait()
        print("Process returncode = " + str(returncode))
        if assert_returncode_zero:
            assert returncode == 0, "Process exited with a non-zero exit code {}; output:{}".format(
                returncode, "".join(self.output)
            )
        return returncode

    def get_execution_time(self):
        return self.exec_time


def create_command_line(args: Dict[str, Any], executable: str) -> str:
    cli_args = " ".join(key if (val is None or val is True) else "{} {}".format(key, val) for key, val in args.items())
    return f"{sys.executable} {executable} {cli_args}"

CACHE_DIR = Path('cache')
LORA_PY = CACHE_DIR.absolute().parent / 'lora.py'
GENAI_DIR = CACHE_DIR.absolute().parent.parent / 'openvino.genai' / 'llm_bench' / 'python'
CONVERT_PY = GENAI_DIR / 'convert.py'
BENCH_PY = GENAI_DIR / 'benchmark.py'

def convert_and_benchmark(exp_dir, lora_torch_dir, group_size, no_ignored_scope):
    print(f'Converting Torch LoRA model to OpenVINO IR in {exp_dir.absolute()}\n\n')
    convert_py_args = {
        "--model_id": lora_torch_dir,
        "--output_dir": exp_dir,
    }
    runner = Command(create_command_line(convert_py_args, CONVERT_PY))
    runner.run()

    ov_model_dir = exp_dir / 'pytorch' / 'dldt' / 'FP32'
    ov_file_path = ov_model_dir / 'openvino_model.xml'
    print(f'Weight Compression to 4bit to {ov_file_path.absolute()}\n\n')
    ov_model = ov.Core().read_model(ov_file_path)
    ignored_scope = None if no_ignored_scope else IgnoredScope(patterns=[".*lora.*"])
    compressed_model = compress_weights(ov_model, mode=CompressWeightsMode.INT4_SYM, ratio=1, group_size=group_size, ignored_scope=ignored_scope)
    # ov_file_path.with_suffix('.bin').unlink()
    # ov_file_path.unlink()
    ov.save_model(compressed_model, ov_file_path)

    print(f'Benchmarking OpenVINO IR in {ov_model_dir.absolute()}\n\n')
    bench_py_args = {
        "-m": ov_model_dir,
        "-p": "\"What is openvino?\"",
        "-n": "2"
    }
    runner = Command(create_command_line(bench_py_args, BENCH_PY))
    runner.run()

    # bin_file = ov_model_dir / 'openvino_model.bin'
    # bin_file.unlink()

def parse_log(path):
    with open(path) as f:
        for line in f.readlines():
            if '[ INFO ] [Average]' in line:
                latency = line.split('Latency: ')[1].split(' ms/token')[0]
                print(latency)

@dataclass
class ExpDesc:
    model_id: str = 'facebook/opt-125m'
    layers: List[str] = None
    rank: int = 8
    group_size: int = 128


MODEL_IDS = [
    # ('facebook/opt-125m', 64),
    ("tinyllama/tinyllama-1.1b-step-50k-105b", 64),
    # ("meta-llama/Llama-2-7b-chat-hf", 128),
]

LAYERS = [
    # ["fc2"],
    # ["fc2", "fc1"],
    # ['down_proj'],
    # ['down_proj', 'o_proj'],
    ['down_proj', 'o_proj', 'up_proj'],
    # ['down_proj', 'o_proj', 'up_proj', 'gate_proj']
]
RANKS = [
    # 4,
    # 8,
    16,
    # 64,
    # 256,
]

EXP_DESCS = [ExpDesc(model_id, layers, rank, group_size) for model_id, group_size in MODEL_IDS for rank in RANKS for layers in LAYERS]

for model_id, group_size in MODEL_IDS:
    try:
        model_name = Path(model_id).name
        exp_name = 'nf4_torch'
        exp_dir = CACHE_DIR / model_name / exp_name

        exp_dir.mkdir(exist_ok=True, parents=True)
        log_path = exp_dir / 'log.txt'
        print('Log file: ', log_path.absolute())
        with log_path.open('a') as f, redirect_stdout(f), redirect_stderr(f):
            print(f'Create FP32 model for model_id={model_id}\n\n')
            lora_torch_dir = CACHE_DIR / model_name / 'nf4_torch'
            if lora_torch_dir.exists():
                shutil.rmtree(lora_torch_dir)
            lora_py_args = {
                "-m": model_id,
                "-o": lora_torch_dir,
                "--fp32": None,
            }
            runner = Command(create_command_line(lora_py_args, LORA_PY))
            runner.run()

            # convert_and_benchmark(exp_dir, lora_torch_dir, group_size, True)
        # parse_log(log_path)
    except Exception as error:
        print("Experiment failed:", error)
        continue


for desc in EXP_DESCS:
    try:
        model_id = desc.model_id
        layers = desc.layers
        rank = desc.rank
        model_name = Path(model_id).name
        layers_str = '_'.join(layers)
        exp_name = f'loftq_{layers_str}_r{rank}'
        exp_dir = CACHE_DIR / model_name / exp_name

        exp_dir.mkdir(exist_ok=True, parents=True)
        log_path = exp_dir / 'log.txt'
        print('Log file: ', log_path)
        with log_path.open('a') as f, redirect_stdout(f), redirect_stderr(f):
            print(f'Create LoRA model for model_id={model_id}\n\n')
            lora_torch_dir = CACHE_DIR / model_name / 'nf4_torch_loftq'
            lora_py_args = {
                "-m": model_id,
                "-o": lora_torch_dir,
                "-l": ' '.join(layers),
                "-r": rank
            }
            runner = Command(create_command_line(lora_py_args, LORA_PY))
            runner.run()

        #     convert_and_benchmark(exp_dir, lora_torch_dir, desc.group_size, False)
        # parse_log(log_path)
    except Exception as error:
        print("Experiment failed:", error)
        continue
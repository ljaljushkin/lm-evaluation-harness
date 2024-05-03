# %%
import os
import json
from itertools import chain
from typing import Any, Dict, Iterable, Optional, Sequence
import traceback

# import pyreft
import torch
import torch.nn as nn
from tqdm import trange
from tqdm.auto import trange
from transformers import PreTrainedModel
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple
import torch.nn.functional as F

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
from pathlib import Path

import torch
from transformers import OPTForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
# from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
from peft import LoftQConfig, LoraConfig, get_peft_model, PeftModel, replace_lora_weights_loftq, TaskType, prepare_model_for_kbit_training
# %%
import os
import random
from typing import Optional

import datasets
import numpy as np
import torch
from datasets import load_dataset
from packaging import version
from tqdm import trange
from transformers import AutoTokenizer, LlamaTokenizer
import matplotlib
matplotlib.use('Agg')

LOGS_DIR = Path("./logs_compress")

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
        # if is_windows():
        #     self.kwargs.update(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
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


def create_command_line(args: Dict[str, Any]) -> str:
    cli_args = " ".join(key if (val is None or val is True) else "{} {}".format(key, val) for key, val in args.items())
    return f"lm_eval {cli_args}"


def main():
    # stabilityai/stablelm-2-zephyr-1_6b
    # mistralai/Mistral-7B-v0.1
    # meta-llama/Meta-Llama-3-8B-Instruct
    # stabilityai/stablelm-tuned-alpha-7b

    model_id = 'microsoft/Phi-3-mini-4k-instruct'
    model_args = f'pretrained={model_id}'

    tuned_adapters_dir = Path('/home/nlyaly/projects/lm-eval-2/cache/Phi-3-mini-4k-instruct')
    DIRS = [
        '10.32_opt_search_q1_wikitext2_loftq_init_R8_Ldug',
        # '10.36_opt_search_q1_wikitext2_loftq_init_R8_Ldugqkvo',
        # '10.39_opt_search_q1_wikitext2_loftq_init_R8_Ldugqkvo'
    ]
    NUM_LAYERS = 32
    limit = 10
    for exp_dir in DIRS:
        all_results_paths = []
        ppls = []
        adapters_dir = tuned_adapters_dir / exp_dir
        for idx in range(NUM_LAYERS):
            task_name = 'wikitext'
            adapter_dir = adapters_dir / str(idx)
            metrics = []
            # log_dir = Path('cache') / model_name / 'int4_via_nf4'
            # log_dir = Path('cache') / model_name / 'fp16'
            log_dir = adapter_dir
            log_dir.mkdir(exist_ok=True, parents=True)
            try:
                print(f"Started experiment on {task_name} in the dir={adapter_dir}\n")
                time_dict = {}
                start_time = time.time()

                results_file = adapter_dir / f'results_{task_name}_l{limit}.json'
                if results_file.exists():
                    results_file.unlink()
                cli_args = {
                    "--model": "hf ",
                    "--model_args":  f"pretrained={model_id},trust_remote_code=True,load_in_8bit=False,dtype=float16,peft_dir={adapter_dir}", #,device=cuda:2
                    "--tasks": task_name,
                    "--limit": limit,
                    '--output_path': results_file,
                }
                runner = Command(create_command_line(cli_args))
                runner.run()
                eval_time = time.time() - start_time
                print(f'eval took {eval_time} seconds')
                print(results_file)
                all_results_paths.append(results_file.resolve())
            except Exception as error:
                print(traceback.print_exc())
                continue
            finally:
                with results_file.open('r') as f:
                    results = json.load(f)
                    word_ppl = results["results"][task_name]["word_perplexity,none"]
                ppls.append(word_ppl)

        print(ppls)
        plt.grid(axis='both', linestyle='-')
        xx = list(range(NUM_LAYERS))
        plt.xticks(xx)
        plt.plot(xx, ppls, **{'marker': 'o'}, label='')
        path = adapters_dir / f'ppls_l{limit}.png'
        plt.savefig(path)
        print('Plotting to ', path)
        plt.clf()
if __name__ == "__main__":
    main()

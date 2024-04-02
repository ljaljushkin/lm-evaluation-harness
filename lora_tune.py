# %%
import os
import time
import json
from argparse import Namespace
from itertools import chain
from typing import Any, Dict, Iterable, Optional, Sequence
import traceback

import torch
import torch.nn as nn
from tqdm import trange
from tqdm.auto import trange
from transformers import PreTrainedModel

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
import openvino as ov
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
try:
    import wandb
    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False
from lion_pytorch import Lion

# EXP_NAME = 'loftq_mse_fp32ref'
# EXP_NAME = 'lora_mse'
EXP_NAME = 'loftq_mse_init_int4'


init_from_scratch = True
MSE_LOFTQ_INIT = True
# optimizer = 'AdamW'
optimizer = 'Lion'
IS_LR_ANNEALING = False
finetune_lr = 1e-3
finetune_relative_mse_tolerance = 0.01
LORA_RANKS = [
    64,
    # 8,
    # 16,
    # 256
]
LORA_LAYERS = [
    # ['down_proj', 'o_proj', 'up_proj', 'gate_proj', 'q_proj', 'k_proj', 'v_proj'],
    # ['down_proj'],
    ['down_proj', 'o_proj'],
    # ['down_proj', 'o_proj', 'up_proj'],
]

# NOTE: with IS_ATTN_MASK_FP32
MODEL_IDS = [
    ('stabilityai/stablelm-2-zephyr-1_6b', False),
    # 'stabilityai/stablelm-3b-4e1t',
    # 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    # '/home/nlyaly/projects/lm-evaluation-harness/cache/stable-zephyr-3b-dpo',
    # ('stabilityai/stablelm-zephyr-3b', False),
    # 'meta-llama/Llama-2-7b-chat-hf',
    # 'HuggingFaceH4/zephyr-7b-beta',
]


bnb_4bit_quant_type = 'nf4'
CACHE_DIR = Path('cache')
BENCH_FILE = CACHE_DIR.parent / 'main.py'
LORA_INIT_DIR = 'cached_init'

TUNE_IDS = None
# NOT_TUNE_IDS = [1, 5, 16]
NOT_TUNE_IDS = []
# TUNE_IDS = [0, 1, 4, 7, 8, 16, 22, 23]
dataset = 'ptb'
nsamples = 64 # TODO: 1024
seed = 0

seqlen = 512
fp32_device = torch.device('cuda:1')
nf4_device = torch.device('cuda:0')
# fp32_device = torch.device('cpu')
# nf4_device = torch.device('cpu')

finetune_adam_beta1 = 0.9
finetune_adam_beta2 = 0.95
finetune_batch_size = 16

# finetune_relative_mse_tolerance = 0.001 # default
local_batch_size = None # 1 # TODO: ???
finetune_max_epochs = 10#000
print_frequency = 10


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


def create_command_line(args: Dict[str, Any], executable: str) -> str:
    cli_args = " ".join(key if (val is None or val is True) else "{} {}".format(key, val) for key, val in args.items())
    return f"{sys.executable} {executable} {cli_args}"

# %%
import contextlib
@contextlib.contextmanager
def using_tf32(enabled: bool):
    was_cudnn = torch.backends.cudnn.allow_tf32
    was_matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = enabled
    torch.backends.cuda.matmul.allow_tf32 = enabled
    yield
    torch.backends.cudnn.allow_tf32 = was_cudnn
    torch.backends.cuda.matmul.allow_tf32 = was_matmul

nf4_model_type = torch.bfloat16
def load_quantized_model_old():
    base_model_dir = '/home/nlyaly/projects/lm-evaluation-harness/cache/stablelm-2-zephyr-1_6b/nf4_torch_loftq'
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=nf4_model_type,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=nf4_model_type,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
        ),
        device_map = nf4_device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        base_model_dir,
        subfolder='loftq_init',
        is_trainable=True,
        device=nf4_device
    )
    model.print_trainable_parameters()
    return model

def get_mae(x, y):
    return (x - y).abs().mean()


def get_mse(x, y):
    return torch.pow(x - y, 2).mean()


def error_report(x, y):
    mae = get_mae(x, y)
    mse = get_mse(x, y)
    print(
        f"Mean absolute error: {mae:>8.5f}\n"
        f"Mean squared error:  {mse:>8.5f}"
    )

current_mse = float("inf")

def load_quantized_model(model_id, model_name, lora_rank_, lora_layers_, init_from_scratch):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=nf4_model_type,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=nf4_model_type,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
        ),
        device_map=nf4_device,
        trust_remote_code=True,
        local_files_only=True,
    )
    loftq_init_dir = CACHE_DIR / model_name / LORA_INIT_DIR
    if init_from_scratch:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=lora_layers_,
            r=lora_rank_,
        )
        model = get_peft_model(base_model, lora_config)

        if MSE_LOFTQ_INIT:
            s = """Beautiful is better than ugly.
                Explicit is better than implicit.
                Simple is better than complex.
                Complex is better than complicated.
                Flat is better than nested.
                Sparse is better than dense.
                Readability counts.
                Special cases aren't special enough to break the rules.
                Although practicality beats purity.
                Errors should never pass silently.
                Unless explicitly silenced.
                In the face of ambiguity, refuse the temptation to guess.
                There should be one-- and preferably only one --obvious way to do it.
                Although that way may not be obvious at first unless you're Dutch.
                Now is better than never.
                Although never is often better than *right* now.
                If the implementation is hard to explain, it's a bad idea.
                If the implementation is easy to explain, it may be a good idea.
                Namespaces are one honking great idea -- let's do more of those!"""

            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                # use_fast=False,
                trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            # TODO:
            """
                ValueError: Asking to pad but the tokenizer does not have a padding token.
                Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)`
                or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.
            """
            loftq_inputs = tokenizer(s.splitlines(), return_tensors="pt", padding=True)
            logits_base = model(**loftq_inputs).logits

            def my_callback(model, module_name):
                """Callable to replace weights with LoFTQ if the mse is lower than the current best one."""
                # TODO: ??
                global current_mse

                logits = model(**loftq_inputs).logits
                mse = get_mse(logits_base, logits)
                if mse < current_mse:
                    current_mse = mse
                    print(f"MSE improved for module {module_name}")
                    return True
                print(f"MSE did not improve for module {module_name}")
                return False
            replace_lora_weights_loftq(model, callback=my_callback)
        # else:
        #     replace_lora_weights_loftq(model)
        # NOTE: double init, one more iteration
        # replace_lora_weights_loftq(peft_model, callback=my_callback)
        model.save_pretrained(loftq_init_dir)
    else:
        model = PeftModel.from_pretrained(
            base_model,
            loftq_init_dir,
            is_trainable=True,
            device=nf4_device
        )
    model.print_trainable_parameters()
    return model

def load_quantized_model_cpu(model_id):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        # device_map = nf4_device,
        use_neural_speed=False,
        trust_remote_code=True,
    )

    # TODO:
    """
            weight = base_layer.weight
        [Previous line repeated 974 more times]
        File "/home/nlyaly/env/lm-eval-fresh/lib/python3.9/site-packages/peft/tuners/tuners_utils.py", line 433, in weight
            base_layer = self.get_base_layer()
        File "/home/nlyaly/env/lm-eval-fresh/lib/python3.9/site-packages/peft/tuners/tuners_utils.py", line 422, in get_base_layer
            while hasattr(base_layer, "base_layer"):
        File "/home/nlyaly/env/lm-eval-fresh/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1695, in __getattr__
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        RecursionError: maximum recursion depth exceeded while calling a Python object
    """
    # loftq_init_dir = CACHE_DIR / MODEL_NAME / LORA_INIT_DIR
    # model = PeftModel.from_pretrained(
    #     base_model,
    #     loftq_init_dir,
    #     is_trainable=True,
    #     device=nf4_device
    # )
    peft_config = LoraConfig(
        r=16,
        task_type=TaskType.CAUSAL_LM,
    )
    model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True
    )
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
# %%
# TODO: FP32 with LoRA or without LoRA?
def load_fp32_lora_model():
    base_model_dir = '/home/nlyaly/projects/lm-evaluation-harness/cache/stablelm-2-zephyr-1_6b/nf4_torch_loftq'
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.bfloat16,
        device_map=nf4_device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        base_model_dir,
        subfolder='loftq_init',
        is_trainable=True,
        device=nf4_device
    )
    model.print_trainable_parameters()
    return model


def set_seed(seed: Optional[int]):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_red_pajama(nsamples, seqlen, tokenizer, eval_mode=False):
    print("Loading red_pajama from togethercomputer/RedPajama-Data-1T-Sample")
    assert not eval_mode, "Only train set is supported in RedPajama"
    traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    trainloader = []
    for _ in trange(nsamples, desc="Making red_pajama calibration set", leave=False):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        assert inp.shape[1] == seqlen
        trainloader.append(inp)
    return trainloader


def get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
    else:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc


def get_ptb(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
        trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
    else:
        valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
    return testenc


def get_c4(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt", max_length=seqlen, truncation=True)
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

    else:
        valdata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        random.seed(0)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            if tmp.input_ids.shape[1] == seqlen:
                # rare case, discovered with Yi tokenizer
                valenc.append(tmp.input_ids)
            else:
                i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                valenc.append(tmp.input_ids[:, i:j])
        valenc = torch.hstack(valenc)
        return valenc


def get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
        trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
    else:
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
        return testenc


def get_c4_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
    else:
        valdata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
        valenc = valenc.input_ids[:, : (256 * seqlen)]
        return valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, eval_mode=False, model_path=None):
    """
    Loads and prepares data for a Transformers model.
    Args:
        name (str): The name of the dataset to load.
        This can be one of 'wikitext2', 'c4', 'ptb','pajama' for datasets loaded from Huggingface datasets,
        or 'none' for cases where a dataset is not needed, like RTN. It can also accept data path to custom file.
        nsamples (int, optional): The number of samples to load from the dataset. Defaults to 128.
        seed (int, optional): The random seed value for data shuffling and splitting. Defaults to 0.
        seqlen (int, optional): The maximum sequence length for input tokenization. Defaults to 2048.
        model_path (str, optional): The path to the pretrained model weights or full model name.
            used to detect llama to call proper tokenizer.
            see https://github.com/huggingface/transformers/issues/22222#issuecomment-1488578722 for reasons.
        eval_mode (bool, optional). defines slice selection for 'wikitext2', 'c4', 'ptb' datasets.
        leave False for train slice.
    Returns:
        data (torch.utils.data.DataLoader or iterable): Data iterable for the dataset.
    Note:
        the popular decapoda-research Llama models have errors in tokenizer config, specifically
        incorrect token ids for BOS, EOS. This gets corrected to ensure compatibility with transformers
        of versions 4.29 and above.
    """
    set_seed(seed)

    # for pre-tokenized datasets

    if name.lower() == "none":
        print("Not loading any dataset. (OK if you use no compression or methods like RTN.)")
        return None
    elif os.path.isfile(name):
        try:
            data = torch.load(name)[:nsamples]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Failed to load custom data from {name}.",
                "Check data path or use one of [c4, wikitext2, ptb, pajama, none]",
            )
    else:
        # for datasets requiring tokenization
        if "llama" in model_path.lower():
            tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)

            # fix for transformer 4.28.0.dev0 compatibility
            if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
                try:
                    tokenizer.bos_token_id = 1
                    tokenizer.eos_token_id = 2
                    print(f"bos/eos tokens updated: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
                except AttributeError:
                    pass
                    print(f"bos/eos tokens unchanged: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                # use_fast=False,
                trust_remote_code=True
            )
        tokenizer.model_max_length = seqlen
        if name.lower() == "wikitext2":
            data = get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "pajama":
            data = get_red_pajama(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "ptb":
            data = get_ptb(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "ptb_new":
            data = get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "c4":
            data = get_c4(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "c4_new":
            data = get_c4_new(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        else:
            raise ValueError(
                f"Failed to load data from {name}.",
                "Check dataset name or path or use one of [c4, wikitext2, ptb, pajama, none]",
            )

    if hasattr(data, "input_ids"):
        data = data.input_ids

    print(f"Loaded data from {name}; {len(data)=} sequences")
    return data

# %%
@torch.no_grad()
def get_inps(
    model, data_iterable, seqlen, nsamples, devices=[fp32_device]
) -> Sequence[torch.Tensor]:
    """
    mocks model launch to collect inputs to the first model layer
    :returns: a list of torch tensors with activations for each device in devices.
    Each tensor has shape [nsample_per_device, seq_len, hid_size]
    """
    offload_activations = False
    print("catching layer inputs from data", flush=True)

    layers = model.model.layers#get_layers(model)

    # nsamples = nsamples or args.nsamples or len(data_iterable)
    device = devices[0] if not offload_activations else torch.device("cpu")
    assert nsamples is not None

    if isinstance(data_iterable, torch.Tensor):
        def batch_generator(testenc, seqlen, nsamples):
            for i in range(nsamples):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(device)
                yield batch

        data_iterable = batch_generator(data_iterable, seqlen, nsamples)

    emb = model.get_input_embeddings()
    emb_device = emb.weight.device
    if emb_device.type != "cuda":
        emb = emb.to(device)
        # opt has other embeddings
        # if model.config.model_type == "opt":
        #     model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
        #     if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        #         model.model.decoder.project_in = model.model.decoder.project_in.to(device)
    device = emb.weight.device  # now default device is the one where the embeddings are.
    layer_device = next(layers[0].parameters()).device
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    nsamples_per_device = (nsamples - 1) // len(devices) + 1
    inps = [
        torch.zeros(
            (min(nsamples_per_device, nsamples - i * nsamples_per_device), seqlen, model.config.hidden_size),
            dtype=dtype,
            device=devices[i] if not offload_activations else "cpu",
            pin_memory=offload_activations,
        )
        for i in range(len(devices))
    ]
    forward_arg_names = ["attention_mask", "position_ids", "use_cache"]
    # if model.config.model_type.lower() in FALCON_TYPES:
    #     forward_arg_names.append("alibi")

    cache = {"i": 0, "alibi": None}

    class CatcherExit(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        @property
        def self_attn(self):
            return self.module.self_attn

        def forward(self, inp, **kwargs):
            inps[cache["i"] // nsamples_per_device][cache["i"] % nsamples_per_device] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise CatcherExit()

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch_inps in data_iterable:
        try:
            if isinstance(batch_inps, (list, tuple)):
                batch_inps, *_ = batch_inps
            batch_inps = batch_inps.to(device)
            # call model.forward to trigger the Catcher
            model(batch_inps, attention_mask=torch.ones_like(batch_inps).to(device))
        except CatcherExit:
            pass  # exit after catcher finished without running the rest of the model layers
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_device)
    model.get_input_embeddings().to(emb_device)
    # if model.config.model_type == "opt":
    #     model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_device)
    #     if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
    #         model.model.decoder.project_in = model.model.decoder.project_in.to(emb_device)
    # NOTE: cpu
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    assert cache["i"] == nsamples
    return inps, forward_args

from typing import Callable, Iterator, Optional, Sequence
def iterate_minibatches(
    *tensors: torch.Tensor,
    batch_size: int,
    allow_incomplete: bool = True,
    device: Optional[torch.device] = None,
    callback: Callable[[Sequence[torch.Tensor]], Sequence[torch.Tensor]] = lambda x: x,
) -> Iterator[Sequence[torch.Tensor]]:
    """
    Samples data points *forever*, in random order, with less overhead than DataLoader;
    Adapted from https://github.com/stanis-morozov/unq/blob/master/lib/utils.py
    probably implemented over9000 times in transformers, torch, etc
    :param tensors: one or more tensors with the same 0-th dimension
    :param batch_size: sample this many points with each yield
    :param allow_incomplete: if True and if dataset size is not divisible by batch size, the last batch
        may have less than :batch_size: samples to cover the entire dataset. If False, the last batch is dropped
    :param callback: optional function to be called on each batch of tensors before it is yielded to the user
    :returns: generates a tuple of minibatches from each tensor, same length as input *tensors
        If a batch contains only one tensor, this function will yield a tensor (and not a tuple/list with one tensor)
    """
    num_samples = len(tensors[0])
    assert all(len(x) == num_samples for x in tensors)
    indices = torch.randperm(num_samples, device=tensors[0].device)
    while True:
        prev_batch = None
        for batch_start in range(0, len(indices), batch_size):
            if not allow_incomplete and batch_start + batch_size > len(indices):
                break
            batch_ix = indices[batch_start : batch_start + batch_size]
            batch = callback(tuple(tensor[batch_ix].to(device, non_blocking=True) for tensor in tensors))
            if prev_batch is not None:
                yield prev_batch
            prev_batch = batch if isinstance(batch, (list, tuple)) and len(tensors) > 1 else batch[0]
            del batch
        yield prev_batch

# %%
@torch.enable_grad()
def finetune_groupwise(
    *,
    layer: nn.Module,
    inp: torch.Tensor,
    out: torch.Tensor,
    device,
    offload_activations = False,
    verbose: bool = True,
    layer_index,
    IS_ATTN_MASK_FP32,
    **kwargs,
) -> nn.Module:
    """
    Fine-tune a module with pre-quantized linear layers so as to minimize MSE between layer-wise inps/outs

    :param layer: a trainable module where linear layers are replaced by QuantizedLinear instances
    :param inp: a list of tensors of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs: a list of tensors of previous output activations, [nsamples_per_device, seq_len, hidden_size]
    :param args: quantization hyperparameters from main.py
    :param kwargs: additional keyword arguments to be passed into layer on each forward
    """
    # lr=1e-4

    assert isinstance(device, torch.device)
    assert isinstance(inp, torch.Tensor) and isinstance(out, torch.Tensor)
    if not offload_activations:
        assert inp.device == out.device == device, (inp.device, out.device, device)
    else:
        assert inp.device == out.device == torch.device("cpu")
        assert inp.is_pinned() and out.is_pinned()

    # replicate non-trainable parameters to each GPU
    replicas = kwargs_by_device = None

    # initialize trainable parameters on main device; prepare to send them to replicas
    differentiable_parameters_by_name = {name: param for name, param in layer.named_parameters() if param.requires_grad}
    param_names, differentiable_parameters = zip(*differentiable_parameters_by_name.items())
    differentiable_parameters = nn.ParameterList(differentiable_parameters)
    # TODO: why zero out only once?
    for param in differentiable_parameters:
        param.grad = torch.zeros_like(param)


    print(f"Fine-tuning {sum(param.numel() for param in differentiable_parameters)} parameters")



    current_lr = finetune_lr
    num_samples_per_device = len(inp[0])
    local_batch_size = None#local_batch_size
    if local_batch_size is None:
        local_batch_size = finetune_batch_size # // len(devices)

    assert all(len(inp_tensor) == num_samples_per_device for inp_tensor in inp)
    assert finetune_batch_size % local_batch_size == 0, ""
    num_accumulation_steps = finetune_batch_size // local_batch_size
    assert num_samples_per_device % local_batch_size * num_accumulation_steps == 0, (
        num_samples_per_device,
        local_batch_size,
    )
    steps_per_epoch = num_samples_per_device // finetune_batch_size

    opt_fn = Lion if optimizer == 'Lion' else torch.optim.AdamW
    opt = opt_fn(differentiable_parameters, lr=finetune_lr, betas=(finetune_adam_beta1, finetune_adam_beta2))
    # const_scheduler = torch.optim.lr_scheduler.ConstantLR(opt, factor=1)
    # scheduler = const_scheduler
    anneal_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_per_epoch // num_accumulation_steps * finetune_max_epochs)
    scheduler = anneal_scheduler
    # scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[const_scheduler, anneal_scheduler], milestones=[steps_per_epoch // num_accumulation_steps])

    batch_iterators = [iterate_minibatches(inp, out, batch_size=local_batch_size, device=device)]

    previous_best_loss = float("inf")  # for early stopping
    steps_accumulated = 0

    loss_name = f"loss_{layer_index}"
    lr_name = f"lr_{layer_index}"

    # define our custom x axis metric
    wandb.define_metric(loss_name)
    # define which metrics will be plotted against it
    wandb.define_metric(lr_name, step_metric=loss_name)

    for epoch in range(finetune_max_epochs):
        loss_numerator = loss_denominator = 0
        for step in range(steps_per_epoch):
            loss = _compute_mse_on_batch(layer, batch_iterators[0], IS_ATTN_MASK_FP32, **kwargs)

            (loss / num_accumulation_steps).backward()
            steps_accumulated += 1

            if not torch.isfinite(loss).item():
                raise ValueError(f"Fine-tuning loss is {loss}")
            if steps_accumulated >= num_accumulation_steps:
                opt.step()
                opt.zero_grad()
                steps_accumulated = 0
                if IS_LR_ANNEALING:
                    scheduler.step()

            loss_numerator += loss.item()
            loss_denominator += 1

            current_loss = loss_numerator / loss_denominator
            log_dict = {
                loss_name: current_loss,
            }
            if IS_LR_ANNEALING:
                current_lr = scheduler.get_last_lr()[0]
                log_dict[lr_name] = current_lr
            current_step = epoch * steps_per_epoch + step
            wandb.log(log_dict)#, step=current_step)
            if verbose and current_step % print_frequency == 0:
                print(f"epoch={epoch}\tstep={step}\tlr={current_lr}\tloss={current_loss:.10f}\t")


        if finetune_relative_mse_tolerance is not None:
            epoch_loss = loss_numerator / loss_denominator
            # TODO: probably need to change threshold as well??
            if epoch_loss / previous_best_loss > (1.0 - finetune_relative_mse_tolerance):
                # NOTE: some straightforward annealing schedule
                # if IS_LR_ANNEALING and current_lr >= 1e-7:
                #     current_lr /= 10
                #     for g in opt.param_groups:
                #         g['lr'] = current_lr
                # else:
                return layer  # early stopping; no updates after last epoch's beam search


            previous_best_loss = min(epoch_loss, previous_best_loss)
    opt.zero_grad(set_to_none=True)
    return layer


def _compute_mse_on_batch(
    layer: nn.Module, batch_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]], IS_ATTN_MASK_FP32, **kwargs
) -> torch.Tensor:
    """
    Compute the activation MSE error between transformer layers
    :param
    """
    inps_batch, outs_batch = next(batch_iter)
    inps_batch = inps_batch.to(dtype=torch.float32)
    outs_batch = outs_batch.to(dtype=torch.float32)
    if IS_ATTN_MASK_FP32:
        for name, value in list(kwargs.items()):
            if isinstance(value, torch.Tensor):
                kwargs[name] = value.to(dtype=torch.float32)

    if inps_batch.shape[0] != 1:  # replicate kwargs to match the batch size
        for name, value in list(kwargs.items()):
            if isinstance(value, torch.Tensor) and value.shape[0] == 1:
                if name not in ("attention_mask", "position_ids"):
                    warnings.warn(f"Tiling an unexpected kwarg {name} over batch size; make sure this is valid.")
                repeats = [len(inps_batch)] + [1 for _ in range(value.ndim - 1)]
                kwargs[name] = value.tile(*repeats)

    outs_prediction, *_unused = layer(inps_batch, **kwargs)
    assert outs_prediction.shape == outs_batch.shape
    return F.mse_loss(outs_prediction, outs_batch)

@torch.no_grad()
def update_outs(
    layer: nn.Module, inps_tensor: torch.Tensor, outs_tensor: torch.Tensor, compute_mse: bool, device, **forward_args
) -> Sequence[float]:
    """
    Update outs_tensor with new activations and optionally compute sample-wise mse loss with previous activations
    :param layer: transformer layer with one or more linear layer to be quantized
    :param inps_tensor: a tensor of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs_tensor: a tensor to write output activations into, [nsamples_per_device, seq_len, hidden_size]
    :note: outs_tensor must contain previous activations with which to compute MSE loss
    :param compute_mse: if True, return a list of sample-wise mse losses; if False, return an empty sequence
    :param forward_args: additional keyword arguments, e.g. attention mask
    :returns: a list of mean squared errors for each sequence
    """
    # device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    out_losses = []
    for j in trange(len(inps_tensor), desc="calc outs after quantization", leave=False):
        outs_batch = layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)[0]
        if compute_mse:
            outs_batch_loss = (
                (outs_batch - outs_tensor[j].to(device))
                .float()
                .square()
                .view(outs_batch.shape[0], -1)
                .mean(dim=1)
                .sqrt()
            )
            outs_batch_loss /= outs_batch.view(outs_batch.shape[0], -1).float().std(dim=1)
            out_losses.append(outs_batch_loss.item())
        outs_tensor[j].copy_(outs_batch.reshape_as(outs_tensor[j]), non_blocking=True)
    return out_losses

# TODO: for FP32 models, for NF4 need extra model!!
# TODO: model specific!
def get_model_head(model):
    head = torch.nn.ModuleList()
    if model.model.norm is not None:
        head.append(model.model.norm)
    head.append(model.lm_head)
    return head

# TODO: model specific!
def get_lm_logits(inps_, model):
    hidden_states = inps_.unsqueeze(0)
    if model.model.norm is not None:
        hidden_states = model.model.norm(hidden_states)
    lm_logits = model.lm_head(hidden_states)
    return lm_logits

@torch.no_grad()
def get_layer_out(layer, inp, forward_args, device):
    for k, v in forward_args.items():
        forward_args[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    out = torch.zeros_like(inp, pin_memory=inp.is_pinned())
    with torch.no_grad():
        # TODO: why not just call layer with the whole batch??
        # because of batch collection in get_inp(), forward_args is with batch=1
        for j in trange(len(inp), desc="calc outs after quantization", leave=False):
            outs_batch = layer(inp[j].to(device).unsqueeze(0), **forward_args)[0]
            out[j].copy_(outs_batch.reshape_as(out[j]), non_blocking=True)
    return out


for (MODEL_ID, IS_ATTN_MASK_FP32) in MODEL_IDS:
    MODEL_NAME = Path(MODEL_ID).name
    for lora_layers in LORA_LAYERS:
        short_layers = ''.join(layer[0] for layer in lora_layers)
        for lora_rank in LORA_RANKS:
            exp_folder =  EXP_NAME + f'_R{lora_rank}_L{short_layers}'
            tuned_model_dir = CACHE_DIR / MODEL_NAME / exp_folder
            print('Experiment dir: ', tuned_model_dir)
            try:
                wandb_run = wandb.init(
                    project="lora_tune",
                    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                    name= exp_folder,
                    # Track hyperparameters and run metadata
                    config={
                        'model_id': MODEL_NAME,
                        "init_from_scratch": init_from_scratch,
                        "mse_loftq_init": MSE_LOFTQ_INIT,
                        "lora_layers": lora_layers,
                        "tune_ids": TUNE_IDS,
                        "rank": lora_rank,
                        "nsamples": nsamples,
                        "dataset": dataset,
                        "seqlen": seqlen,
                        'optimizer': optimizer,
                        "bnb_4bit_quant_type": bnb_4bit_quant_type,
                        "finetune_lr": finetune_lr,
                        "finetune_adam_beta1": finetune_adam_beta1,
                        "finetune_adam_beta2": finetune_adam_beta2,
                        "finetune_batch_size": finetune_batch_size,
                        "finetune_relative_mse_tolerance": finetune_relative_mse_tolerance,
                        "local_batch_size": local_batch_size,
                        "finetune_max_epochs": finetune_max_epochs,
                        "tuned_model_dir": tuned_model_dir,
                        "is_lr_annealing": IS_LR_ANNEALING,
                    }
                )

                # %%
                fp32_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    device_map=fp32_device,
                    trust_remote_code=True,
                )
                # fp32_model.save_pretrained('/home/nlyaly/projects/lm-evaluation-harness/cache/stablelm-2-zephyr-1_6b/fp32')
                # assert False
                nf4_model = load_quantized_model(MODEL_ID, MODEL_NAME, lora_rank, lora_layers, init_from_scratch=init_from_scratch)
                NUM_LAYERS = nf4_model.config.num_hidden_layers
                if TUNE_IDS is None:
                    TUNE_IDS = list(range(NUM_LAYERS))
                # nf4_model = load_fp32_lora_model()
                # nf4_model = load_quantized_model_cpu()

                # %%
                dataloader = get_loaders(
                    dataset,
                    nsamples=nsamples,
                    seed=seed,
                    model_path=MODEL_ID,
                    seqlen=seqlen,
                )


                # %%
                inps_tensor, forward_args = get_inps(fp32_model, dataloader, seqlen, nsamples)

                # %%
                fp32_layers = fp32_model.model.layers
                fp32_inp = inps_tensor[0]
                nf4_inp = fp32_inp.clone()


                assert fp32_inp.shape == nf4_inp.shape == fp32_inp.shape
                total_layer_time = 0

                for layer_index in range(len(fp32_layers)):
                    print(f"\n---------------- Layer {layer_index} of {len(fp32_layers)} ----------------")
                    stats_payload = {}
                    start_time = time.time()

                    fp32_layer = fp32_layers[layer_index]
                    # NOTE: recalculation of fp32 output for tuning
                    fp32_out = get_layer_out(fp32_layer, fp32_inp, forward_args, fp32_device)

                    fp32_out = fp32_out.to(nf4_device)
                    fp32_inp = fp32_inp.to(nf4_device)
                    nf4_inp = nf4_inp.to(nf4_device)

                    nf4_layer = nf4_model.model.model.layers[layer_index]

                    layer_dtype_original = next(nf4_layer.parameters()).dtype
                    # TODO: is bfloat16 to tf32 needed for NF4 model???
                    # otherwise the error happened
                    #   return torch.layer_norm(input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)
                    #   RuntimeError: expected scalar type Float but found BFloat16
                    nf4_layer = nf4_layer.to(dtype=torch.float32)
                    with using_tf32(enabled=True):
                        for k, v in forward_args.items():
                            forward_args[k] = v.to(nf4_device) if isinstance(v, torch.Tensor) else v
                    # NOTE: FP32 REF
                    # layer = finetune_groupwise(layer=nf4_layer, inp=nf4_inp, out=fp32_out, **forward_args, device=nf4_device)
                    # NOTE: NF4 REF
                    # TODO: think about criteria for tuning!!!
                    if layer_index in TUNE_IDS and layer_index not in NOT_TUNE_IDS:
                        nf4_layer = finetune_groupwise(layer=nf4_layer, inp=fp32_inp, out=fp32_out, **forward_args, device=nf4_device, layer_index=layer_index, IS_ATTN_MASK_FP32=IS_ATTN_MASK_FP32)
                    nf4_layer = nf4_layer.to(dtype=nf4_model_type)

                    layer_dir = tuned_model_dir / str(layer_index)
                    print(f'saving to tuned adapters for {layer_index} layer in {layer_dir}')
                    nf4_model.save_pretrained(layer_dir)

                    # ============prepare inputs for next iteration===============
                    # NOTE: FP32 REF - why is it working so bad??
                    # # override fp32 input by fp32 output for next iteration
                    # fp32_inp.copy_(fp32_out, non_blocking=True)
                    # # calculate output for (nf4 + tuned) layer given nf4 input
                    # # compare with fp32 output and copy result to fp32 output
                    # out_losses = update_outs(nf4_layer, nf4_inp, fp32_out, compute_mse=True, **forward_args, device=nf4_device)
                    # # override nf4 input by (nf4+tuned) output for next iteration
                    # nf4_inp.copy_(fp32_out, non_blocking=True)

                    # NOTE: FP32 REF
                    out_losses = update_outs(nf4_layer, fp32_inp, fp32_out, compute_mse=True, **forward_args, device=nf4_device)
                    fp32_inp, fp32_out = fp32_out, fp32_inp

                    # NOTE: cpu
                    torch.cuda.empty_cache()
                    # Logging
                    layer_time = time.time() - start_time
                    total_layer_time += layer_time
                    stats_payload["layer_time"] = layer_time

                    stats_payload["out_loss"] = torch.mean(torch.Tensor(out_losses)).item()
                    stats_payload["Step"] = layer_index
                    wandb.log({"out_loss": stats_payload["out_loss"]})
                    wandb.log({"layer_time": stats_payload["layer_time"]})
                    print(stats_payload)

                    # PRINT_IDS = [0, 5, 10, 15, 20, 23]
                    # PRINT_IDS = TUNE_IDS
                    PRINT_IDS = [NUM_LAYERS - 1]
                    if layer_index in PRINT_IDS:
                        print(f'\n\nBenchmarking via lm-eval-harness from folder {layer_dir.absolute()}\n')
                        cli_args = {
                            "--tuned_adapters_dir": layer_dir,
                            "--model": MODEL_ID,
                        }
                        runner = Command(create_command_line(cli_args, BENCH_FILE))
                        runner.run()
                        eval_results_file = layer_dir / 'results.json'
                        with eval_results_file.open('r') as f:
                            j = json.load(f)
                            word_ppl = j["results"]["wikitext"]["word_perplexity"]
                            wandb.log({"word_ppl_wiki": word_ppl})
                print(f'Tuning took: {total_layer_time:.1f} seconds')
                wandb.run.summary["tuning_time"] = total_layer_time
            except Exception as error:
                print(traceback.print_exc())
                print(f"Experiment failed with {error}")
                continue
            finally:
                wandb_run.finish()
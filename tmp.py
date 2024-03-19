# %%
import os
import time
from argparse import Namespace
from itertools import chain
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
from tqdm import trange
from tqdm.auto import trange
from transformers import PreTrainedModel

from typing import Tuple
import torch.nn.functional as F

import torch
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoftQConfig, LoraConfig, get_peft_model, PeftModel
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


dataset = 'c4'
nsamples = 20 # 100 # TODO: 1024
seed = 0
model_path = 'stabilityai/stablelm-2-zephyr-1_6b'
seqlen = 256
fp32_device = torch.device('cuda:1')
nf4_device = torch.device('cuda:0')







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


def load_quantized_model():
    base_model_dir = '/home/nlyaly/projects/lm-evaluation-harness/cache/stablelm-2-zephyr-1_6b/nf4_torch_loftq'
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type='nf4',
        ),
        device_map = nf4_device
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

# %%
# TODO: FP32 with LoRA or without LoRA?
def load_fp32_lora_model():
    base_model_dir = '/home/nlyaly/projects/lm-evaluation-harness/cache/stablelm-2-zephyr-1_6b/nf4_torch_loftq'
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.bfloat16,
        device_map=nf4_device
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
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

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
    model, data_iterable, seqlen, nsamples
) -> Sequence[torch.Tensor]:
    """
    mocks model launch to collect inputs to the first model layer
    :returns: a list of torch tensors with activations for each device in devices.
    Each tensor has shape [nsample_per_device, seq_len, hid_size]
    """
    devices = [fp32_device]
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
            model(batch_inps, attention_mask=torch.ones_like(batch_inps))
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
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    devices,
    offload_activations = False,
    verbose: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Fine-tune a module with pre-quantized linear layers so as to minimize MSE between layer-wise inps/outs

    :param layer: a trainable module where linear layers are replaced by QuantizedLinear instances
    :param inps: a list of tensors of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs: a list of tensors of previous output activations, [nsamples_per_device, seq_len, hidden_size]
    :param args: quantization hyperparameters from main.py
    :param kwargs: additional keyword arguments to be passed into layer on each forward
    """
    # lr =1e-4
    finetune_lr = 1e-5
    finetune_adam_beta1 = 0.9
    finetune_adam_beta2 = 0.95
    finetune_batch_size = 5 # 32
    relative_mse_tolerance = 0.01
    finetune_relative_mse_tolerance = 0.001
    local_batch_size = 1 # TODO: ???
    finetune_max_epochs = 1000
    print_frequency = 1

    assert isinstance(devices, (list, tuple)) and len(devices) >= 1, f"Found devices = {devices}"
    assert isinstance(inps, (list, tuple)) and isinstance(inps, (list, tuple))
    assert len(inps) == len(outs) == len(devices)
    for i in range(len(devices)):
        assert isinstance(inps[i], torch.Tensor) and isinstance(outs[i], torch.Tensor)
        if not offload_activations:
            assert inps[i].device == outs[i].device == devices[i], (inps[i].device, outs[i].device, devices)
        else:
            assert inps[i].device == outs[i].device == torch.device("cpu")
            assert inps[i].is_pinned() and outs[i].is_pinned()

    # replicate non-trainable parameters to each GPU
    replicas = kwargs_by_device = None
    # if len(devices) > 1:
    #     replicas = torch.nn.parallel.replicate(layer, devices)
    #     replicas[0] = layer
    #     kwargs_by_device = []
    #     for device in devices:
    #         kwargs_by_device.append(
    #             {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
    #         )

    # initialize trainable parameters on main device; prepare to send them to replicas
    differentiable_parameters_by_name = {name: param for name, param in layer.named_parameters() if param.requires_grad}
    param_names, differentiable_parameters = zip(*differentiable_parameters_by_name.items())
    differentiable_parameters = nn.ParameterList(differentiable_parameters)
    # TODO: why zero out only once?
    for param in differentiable_parameters:
        param.grad = torch.zeros_like(param)

    # if replicas:
    #     replacement_tables = _make_parameter_replacement_tables(layer, replicas, param_names, differentiable_parameters)

    print(f"Fine-tuning {sum(param.numel() for param in differentiable_parameters)} parameters")
    opt = torch.optim.Adam(
        differentiable_parameters, lr=finetune_lr, betas=(finetune_adam_beta1, finetune_adam_beta2)
    )

    # backup best parameters
    # if args.finetune_keep_best:
    #     best_parameters = deepcopy(differentiable_parameters)

    assert finetune_batch_size % len(devices) == 0, "batch_size must be divisible by the number of GPUs"

    num_samples_per_device = len(inps[0])
    local_batch_size = local_batch_size
    if local_batch_size is None:
        local_batch_size = finetune_batch_size // len(devices)

    assert all(len(inps_tensor) == num_samples_per_device for inps_tensor in inps)
    assert finetune_batch_size % (local_batch_size * len(devices)) == 0, ""
    num_accumulation_steps = finetune_batch_size // (local_batch_size * len(devices))
    assert num_samples_per_device % local_batch_size * num_accumulation_steps == 0, (
        num_samples_per_device,
        local_batch_size,
    )
    steps_per_epoch = num_samples_per_device * len(devices) // finetune_batch_size
    batch_iterators = [
        iterate_minibatches(inps[i], outs[i], batch_size=local_batch_size, device=devices[i])
        for i in range(len(devices))
    ]

    previous_best_loss = float("inf")  # for early stopping
    steps_accumulated = 0
    for epoch in range(finetune_max_epochs):
        loss_numerator = loss_denominator = 0
        for step in range(steps_per_epoch):
            # if len(devices) == 1:
            loss = _compute_mse_on_batch(layer, batch_iterators[0], **kwargs)
            # else:
            #     loss = _compute_mse_parallel(
            #         devices,
            #         replicas,
            #         differentiable_parameters,
            #         replacement_tables,
            #         batch_iterators,
            #         kwargs_by_device,
            #     )

            # retain_graph = False#not(steps_accumulated + 1 >= num_accumulation_steps)
            # TODO: how did it work without retain_graph=True???
            (loss / num_accumulation_steps).backward()
            steps_accumulated += 1

            if not torch.isfinite(loss).item():
                raise ValueError(f"Fine-tuning loss is {loss}")
            # TODO: why doesn't work??? because reference outputs were collected with grad from floating point model!!
            if steps_accumulated >= num_accumulation_steps:
                opt.step()
                # TODO: usually in the beginning
                opt.zero_grad()
                steps_accumulated = 0

            loss_numerator += loss.item()
            loss_denominator += 1
            if verbose and (epoch * steps_per_epoch + step) % print_frequency == 0:
                print(f"epoch={epoch}\tstep={step}\tloss={loss_numerator / loss_denominator:.10f}\t")

        # TODO: why comment?
        # if verbose and (epoch * steps_per_epoch + step) % print_frequency != 0:
        #     print(f"epoch={epoch}\tstep={step}\tloss={loss_numerator / loss_denominator:.10f}\t")

        if finetune_relative_mse_tolerance is not None:
            epoch_loss = loss_numerator / loss_denominator
            # if args.finetune_keep_best:
            #     if epoch_loss / previous_best_loss < 1.0:
            #         best_parameters = deepcopy(differentiable_parameters)
            #     else:
            #         differentiable_parameters = best_parameters
            if epoch_loss / previous_best_loss > (1.0 - finetune_relative_mse_tolerance):
                return layer  # early stopping; no updates after last epoch's beam search
            previous_best_loss = min(epoch_loss, previous_best_loss)
    opt.zero_grad(set_to_none=True)
    return layer


def _compute_mse_on_batch(
    layer: nn.Module, batch_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]], **kwargs
) -> torch.Tensor:
    """
    Compute the activation MSE error between transformer layers
    :param
    """
    inps_batch, outs_batch = next(batch_iter)
    inps_batch = inps_batch.to(dtype=torch.float32)
    outs_batch = outs_batch.to(dtype=torch.float32)

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
    layer: nn.Module, inps_tensor: torch.Tensor, outs_tensor: torch.Tensor, compute_mse: bool, **forward_args
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
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
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

# @torch.no_grad()
# def perplexity_eval(model, testenc, args):
#     dataset_name = ''
#     print(f"\nEvaluating perplexity for {args.dataset_name} dataset ...")

#     nsamples = testenc.numel() // model.seqlen

#     use_cache = model.config.use_cache
#     model.config.use_cache = False

#     inps, forward_args = get_inps(model, testenc, args, nsamples=nsamples)
#     outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]
#     device = args.devices[0]
#     for k, v in forward_args.items():
#         forward_args[k] = v.to(device) if isinstance(v, torch.Tensor) else v

#     layers = get_layers(model)
#     for i in trange(len(layers), desc="processing eval data by layer"):
#         layer = layers[i].to(device)
#         if len(args.devices) == 1:
#             assert len(inps) == len(outs) == 1
#             update_outs(layer, inps[0], outs[0], compute_mse=False, **forward_args)
#         else:
#             update_outs_parallel(args.devices, layer, inps, outs, compute_mse=False, **forward_args)
#         layers[i] = layer.cpu()
#         del layer
#         torch.cuda.empty_cache()
#         inps, outs = outs, inps

#     get_model_head(model).to(device)
#     testenc = testenc.to(device)
#     nsamples_per_device = len(inps[0])
#     assert len(set(map(len, inps[:-1]))) <= 1 and len(inps[-1]) <= len(inps[0])

#     nlls = []
#     for i in range(nsamples):
#         inp = inps[i // nsamples_per_device][i % nsamples_per_device].to(args.devices[0], non_blocking=True)
#         lm_logits = get_lm_logits(inp.to(device), model)
#         shift_logits = lm_logits[:, :-1, :].contiguous()
#         shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
#         loss_fct = nn.CrossEntropyLoss()
#         loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#         neg_log_likelihood = loss.float() * model.seqlen
#         nlls.append(neg_log_likelihood)
#     ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
#     print(f"\n{args.dataset_name} perplexity = {ppl.item():.4f}\n")

#     get_model_head(model).to(torch.device("cpu"))

#     if args.wandb:
#         wandb.log({args.dataset_name: ppl.item()})

#     model.config.use_cache = use_cache










# assert has_wandb, "`wandb` not installed, try pip install `wandb`"
# wandb.init(
#     project="lora_tune",
#     # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
#     name="loftq_debug",
#     # Track hyperparameters and run metadata
#     config={
#         "lora_layers": ['down_proj'],
#         "rank": 64,
#         "iter": 5,
#         "learning_rate": 0.02,
#         "nsamples": nsamples,
#         "dataset": dataset,
#         "seqlen": seqlen,
#     }
# )

# %%
fp32_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=fp32_device
)

nf4_model = load_quantized_model()
# nf4_model = load_fp32_lora_model()

# %%
dataloader = get_loaders(
    dataset,
    nsamples=nsamples,
    seed=seed,
    model_path=model_path,
    seqlen=seqlen,
)

# %%
inps_tensor, forward_args = get_inps(fp32_model, dataloader, seqlen, nsamples)

# %%
fp32_layers = fp32_model.model.layers
fp32_layer = fp32_layers[0]


# %%
inps = inps_tensor
outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]
with torch.no_grad():
    for j in trange(len(inps_tensor[0]), desc="calc outs after quantization", leave=False):
        outs_batch = fp32_layer(inps[0][j].to(fp32_device).unsqueeze(0), **forward_args)[0]
        outs[0][j].copy_(outs_batch.reshape_as(outs[0][j]), non_blocking=True)
outs[0] = outs[0].to(nf4_device)
inps[0] = inps[0].to(nf4_device)
for k, v in forward_args.items():
    forward_args[k] = v.to(nf4_device) if isinstance(v, torch.Tensor) else v
assert len(outs) == len(inps)

for layer_index in range(1):#len(fp32_layers)):
    print(f"\n---------------- Layer {layer_index} of {len(fp32_layers)} ----------------")
    stats_payload = {}
    start_time = time.time()

    nf4_layer = nf4_model.model.model.layers[layer_index]

    nf4_layer = nf4_layer.to(dtype=torch.float32)
    with using_tf32(enabled=True):
        layer = finetune_groupwise(layer=nf4_layer, inps=inps, outs=outs, **forward_args, devices=[nf4_device])

    model_dir = '/home/nlyaly/projects/lm-evaluation-harness/cache/stablelm-2-zephyr-1_6b/nf4_torch_loftq_tuned'
    print(f'saving to {model_dir}')
    nf4_model.save_pretrained(model_dir)
    # if save_dir:
    #     save_dir.mkdirs(exist_ok=True)
    #     layer_save_path = save_dir / f"{layer_index}.pth")
    #     print(f"Saving layer {layer_index}... to {layer_save_path}")
    #     torch.save(layer, layer_save_path)

    # assert len(inps) == len(outs) == 1
    # out_losses = update_outs(layer, inps[0], outs[0], compute_mse=not args.skip_out_loss, **forward_args)

    # torch.cuda.empty_cache()

    # # TODO:!
    # inps, outs = outs, inps

    # Logging
    stats_payload["layer_time"] = time.time() - start_time
    # stats_payload["out_loss"] = torch.mean(torch.Tensor(out_losses)).item()
    stats_payload["Step"] = layer_index
    # if has_wandb:
    #     # wandb.log({"out_loss": stats_payload["out_loss"]}, step=layer_index)
    #     wandb.log({"layer_time": stats_payload["layer_time"]}, step=layer_index)
    print(stats_payload)

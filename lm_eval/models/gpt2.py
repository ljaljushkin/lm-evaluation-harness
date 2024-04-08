import shutil
import torch
import json
import os
import transformers
from pathlib import Path
from typing import Optional, Union
from lm_eval.base import BaseLM
from peft import PeftModel, PeftConfig
from peft import LoftQConfig, LoraConfig, get_peft_model
from safetensors import safe_open
# from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch.nn as nn

class Shell(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)

def unwrap_model(model, sub_module_name=".base_layer"):
    sub_module_name_list = [k.split(sub_module_name)[0] for k in model.state_dict().keys() if sub_module_name in k]
    sub_module_name_set = set(sub_module_name_list)
    for name in sub_module_name_set:
        # get the parent of the submodule
        name_parent = ".".join(name.split(".")[:-1])
        name_child = name.split(".")[-1]
        sub_module = model.get_submodule(name_parent)
        print(sub_module)

        # replace with shell
        child = getattr(sub_module, name_child)
        weight = getattr(child.base_layer, "weight", None)
        bias = getattr(child.base_layer, "bias", None)
        shell = Shell(weight, bias)

        setattr(sub_module, name_child, shell)

    print("You have unwrapped the model. Use it on your own risk.")


def print_model(model, name):
    print("=" * 10 + name + "=" * 10)
    print(model)
    for name, param in model.named_parameters():
        if torch.is_tensor(param):
            if param.dtype in [torch.float32, torch.float16]:
                print(
                    name,
                    param.shape,
                    param.device,
                    param.dtype,
                    param.requires_grad,
                    param.mean().item(),
                    param.max().item(),
                )
            else:
                print(name, param.shape, param.device, param.dtype, param.requires_grad)

def save_loftq_init(pretrained, base_model_dir, loftq_iter, rank, target_modules):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto'
    )

    loftq_config = LoftQConfig(
        loftq_bits=4,
        loftq_iter=loftq_iter
    )
    lora_config = LoraConfig(
        init_lora_weights="loftq",
        loftq_config=loftq_config,
        r=rank,
        target_modules=target_modules,
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model_dir = base_model_dir / 'loftq_init'
    # Save LoftQ model
    lora_model.save_pretrained(lora_model_dir)
    print_model(lora_model, "lora_model")
    base_model = lora_model.get_base_model()

    # remove lora adapters and save the backbone
    unwrap_model(base_model)
    base_model.save_pretrained(base_model_dir)

    print_model(base_model, "base_model")

    # convert safetensor to bin
    tensors = {}
    with safe_open(lora_model_dir / "adapter_model.safetensors", framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    torch.save(tensors, lora_model_dir / "adapter_model.bin")

    # change adapter_config.json
    with open(lora_model_dir / "adapter_config.json", "r") as fp:
        adapter_config = json.load(fp)
        adapter_config['base_model_name_or_path'] = str(base_model_dir)  # This can be a local path or Hub model id
        adapter_config['init_lora_weights'] = True  # Don't apply LoftQ when loading again
        fp.close()
    with open(os.path.join(lora_model_dir, "adapter_config.json"), "w") as fp:
        json.dump(adapter_config, fp, indent=2)

    return lora_model_dir

def _get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class HFLM(BaseLM):

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        device=None,#"cuda",
        pretrained="gpt2",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        max_batch_size=512,
        max_length=None,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        tuned_adapters_dir=None
    ):
        super().__init__()

        # Initialize model
        if isinstance(pretrained, transformers.PreTrainedModel):
            self.model = pretrained
            self._device = self.model.device

            if tokenizer:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
            else:
                # Get tokenizer
                model_name = self.model.name_or_path
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                )

        elif isinstance(pretrained, str):

            # Initialize device
            # assert isinstance(device, str)
            # device_list = set(
            #     ["cuda", "cpu"]
            #     + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            # )
            # if device and device in device_list:
            #     self._device = torch.device(device)
            #     print(f"Using device '{device}'")
            # else:
            #     print("Device not specified")
            #     print(f"Cuda Available? {torch.cuda.is_available()}")
            #     self._device = (
            #         torch.device("cuda")
            #         if torch.cuda.is_available()
            #         else torch.device("cpu")
            #     )
            self._device = 'cuda:2'
            revision = revision + ("/" + subfolder if subfolder is not None else "")

            # TODO: is it really needed?
            # fp32_model_dir = Path('/home/nlyaly/projects/lm-evaluation-harness/cache/stablelm-2-zephyr-1_6b/nf4_torch_loftq')
            # fp32_model_dir = Path('/home/nlyaly/projects/lm-evaluation-harness/cache/stablelm-2-zephyr-1_6b/fp32')
            # adapter_model_dir = Path('/home/nlyaly/projects/lm-evaluation-harness/cache/stablelm-2-zephyr-1_6b/nf4_torch_loftq_tuned_17.69/23')

            # fp32_model_dir = '/home/nlyaly/projects/lm-evaluation-harness/cache/stablelm-2-zephyr-1_6b/fp32'
            # if base_model_dir.exists():
            #     shutil.rmtree(base_model_dir)
            # base_model_dir.mkdir(exist_ok=True, parents=True)

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer if tokenizer else pretrained,
                revision=revision,
                use_fast=True,
                trust_remote_code=trust_remote_code,
            )
            # TODO: is it needed?
            # self.tokenizer.model_max_length = 2048#self.max_length

            # self.tokenizer.save_pretrained(base_model_dir)

            # TODO: Does it quantize/dequantize weights in-place??
            # lora_model_dir = save_loftq_init(
            #     pretrained, base_model_dir,
            #     loftq_iter=5,
            #     rank=64,
            #     target_modules=["down_proj"]#, "o_proj"],#, "up_proj", "gate_proj"]
            # )

            # TODO: evaluate fp32 model
            # self.model = AutoModelForCausalLM.from_pretrained(
            #     fp32_model_dir,
            #     torch_dtype=torch.bfloat16,
            #     device_map=torch.device('cuda:0')
            #     # quantization_config=BitsAndBytesConfig(
            #     #     load_in_4bit=True,
            #     #     bnb_4bit_compute_dtype=torch.bfloat16,
            #     #     bnb_4bit_use_double_quant=False,
            #     #     bnb_4bit_quant_type='nf4',
            #     # ),
            # )

            # TODO: load pure FP32 by model_id
            # NOTE: evaluation of NF4 + adapters model
            base_model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                # quantization_config=BitsAndBytesConfig(
                #     load_in_4bit=True,
                #     bnb_4bit_compute_dtype=torch.bfloat16,
                #     bnb_4bit_use_double_quant=False,
                #     bnb_4bit_quant_type='nf4',
                # ),
                device_map = 'auto',#torch.device('cuda:2')
                trust_remote_code=trust_remote_code,
            )
            self.model = base_model
            # TODO: add adapters and initialize by tuned weights
            # self.model = PeftModel.from_pretrained(
            #     base_model,
            #     tuned_adapters_dir,
            #     # base_model_dir,
            #     # subfolder=lora_model_dir.name,
            #     is_trainable=False,
            #     device_map = torch.device('cuda:2'),
            #     trust_remote_code=trust_remote_code
            # )
            print(base_model)

            # NOTE: SignRound
            # quantized_model_path = "/home/nlyaly/projects/lm-evaluation-harness/tmp_autoround_cpu"
            # config = transformers.AutoConfig.from_pretrained(quantized_model_path)
            # # config.quantization_config["use_exllama"] = True
            # config.quantization_config["disable_exllama"] = True
            # woq_config = WeightOnlyQuantConfig(
            #     group_size=64, scheme="sym", use_autoround=True, compute_dtype="int4_fullrange"
            # )  ##only supports 4 bits currently
            # self.model = AutoModelForCausalLM.from_pretrained(
            #     quantized_model_path, quantization_config=woq_config, trust_remote_code=True,
            #     # config=config,
            #     device_map='cuda'
            # )

            # config = PeftConfig.from_pretrained(pretrained)

            # TODO: pure nf4
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.bfloat16,
            #     bnb_4bit_use_double_quant=False,
            # )

            # bnb_config=BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is recommended
            #     bnb_4bit_use_double_quant=False,
            #     bnb_4bit_quant_type='nf4',
            # )

            # Initialize new model and tokenizer instances
            # self.model = PeftModel.from_pretrained(
            #     self.model,
            #     pretrained
            # )



        else:
            raise TypeError(
                "Parameter pretrained should be of type str or transformers.PreTrainedModel"
            )

        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size

        # Validate batch_size
        assert isinstance(batch_size, (int, str))

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_length = max_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
            generation_kwargs[
                "pad_token_id"
            ] = eos_token_id  # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)


# for backwards compatibility
GPT2LM = HFLM

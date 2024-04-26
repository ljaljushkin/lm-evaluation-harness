from pathlib import Path
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    replace_lora_weights_loftq,
    LoraConfig,
    get_peft_model
)
model_id = '/home/nlyaly/projects/lm-evaluation-harness/cache/stablelm-tuned-alpha-7b/fp32'
nf4_model_type = torch.bfloat16
nf4_device = torch.device('cuda:0')
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=nf4_model_type,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=nf4_model_type,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type='nf4',
    ),
    device_map=nf4_device,
    trust_remote_code=True,
    local_files_only=True,
)
print(base_model)
NUM_LAYERS = base_model.config.num_hidden_layers

LORA_LAYERS = [
    ('dense_h_to_4h', 'dense_4h_to_h'),
    (), # query_key_value
]

#TUNE_IDS = [0]
TUNE_IDS = list(range(NUM_LAYERS))

layer_names = []
for i in TUNE_IDS:
    mlp_layer, attn_layer = LORA_LAYERS
    layer_names.extend(f'layers.{i}.mlp.{name}' for name in mlp_layer)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=layer_names,
    r=8,
)
model = get_peft_model(base_model, lora_config)
replace_lora_weights_loftq(model, model_path=model_id)
model.print_trainable_parameters()

init_dir = Path('/home/nlyaly/projects/lm-evaluation-harness/cache/stablelm-tuned-alpha-7b/loftq_scaled_init')
model.save_pretrained(init_dir)

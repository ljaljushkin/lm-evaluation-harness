from nncf import compress_weights, CompressWeightsMode, IgnoredScope
import openvino as ov
from pathlib import Path
import argparse
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from peft import LoftQConfig, LoraConfig, get_peft_model

model_id = Path('/home/nlyaly/projects/lm-evaluation-harness/cache/opt-125m/lora_torch')
model = OVModelForCausalLM.from_pretrained(
    model_id,
    export=True,
    trust_remote_code=True,
    use_cache=True,
    # ov_config=ov_config,
    stateful=False,
    load_in_8bit=False,
)
# model.save_model()

# int4_model_path = fp32_model_path.parent.parent / 'INT4'
# int4_model_path.mkdir(exist_ok=True)

# ov_model = ov.Core().read_model(fp32_model_path)
# compressed_model = compress_weights(ov_model, mode=CompressWeightsMode.INT4_SYM, ratio=1, group_size=64, ignored_scope=IgnoredScope(patterns=[".*lora_.*"],))
# ov.save_model(compressed_model, int4_model_path /'openvino_model.xml')
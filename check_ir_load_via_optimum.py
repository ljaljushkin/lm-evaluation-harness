from optimum.intel import OVModelForCausalLM
from transformers import AutoModelForCausalLM
from openvino.tools.mo import convert_model
import torch
from openvino.runtime import serialize

MODEL_ID = "openlm-research/open_llama_3b"
# MODEL_ID = "databricks/dolly-v2-3b"

OUT_DIR_FP32 = "runs/debug"
example_input={"input_ids": torch.ones((1,12), dtype=torch.long),"attention_mask": torch.ones((1,12), dtype=torch.long)}

torch_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, use_cache=False, torchscript=True)
# ov_model = convert_model(torch_model, example_input=example_input)
# serialize(ov_model, f"{OUT_DIR_FP32}/openvino_model.xml")
# torch_model.config.to_json_file(f"{OUT_DIR_FP32}/config.json")

# model = OVModelForCausalLM.from_pretrained(f"{OUT_DIR_FP32}", use_cache=False)
# model(example_input)
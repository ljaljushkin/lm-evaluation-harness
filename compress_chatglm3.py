import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, AutoConfig
from optimum.intel import OVModelForCausalLM
from nncf import compress_weights
from nncf import CompressWeightsMode

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", use_fast=True, trust_remote_code=True)
model = OVModelForCausalLM.from_pretrained("THUDM/chatglm3-6b",  use_cache=True,  export=True, trust_remote_code=True, compile=False, load_in_8bit=False)

model.model = compress_weights(model.model, mode=CompressWeightsMode.INT4_SYM, group_size=128, ratio=0.8) # model is openvino.Model object

model.save_pretrained("chatglm3_nncf_ov")
tokenizer.save_pretrained("chatglm3_nncf_ov")
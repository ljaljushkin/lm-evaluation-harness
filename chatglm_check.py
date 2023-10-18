from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
ov_model = OVModelForCausalLM.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, export=True)
tokens = tokenizer("你好", return_tensors="pt")
print(tokens)

logits = ov_model(**tokens).logits
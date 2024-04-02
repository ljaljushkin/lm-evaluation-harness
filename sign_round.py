from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = 'stabilityai/stablelm-2-zephyr-1_6b'
# MODEL_ID = 'TinyLlama/TinyLlama-1.1B-step-50K-105b'
#MODEL_ID = 'stabilityai/stablelm-3b-4e1t'
# MODEL_ID = 'meta-llama/Llama-2-7b-chat-hf'
# MODEL_ID = 'HuggingFaceH4/zephyr-7b-beta'
# MODEL_ID = '/home/nlyaly/projects/lm-evaluation-harness/cache/stable-zephyr-3b-dpo'
# MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
# model_name = "meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", trust_remote_code=True, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

from auto_round import AutoRound

bits, group_size, sym = 4, 64, True
## The device will be detected automatically, or you can specify it.
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym, device='cuda')
autoround.quantize()
print(autoround)
output_dir = "./tmp_autoround_cpu"
autoround.save_quantized(output_dir)

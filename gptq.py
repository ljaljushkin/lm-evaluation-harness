from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch

model_id = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer, use_cuda_fp16=True, batch_size=1)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config, torch_dtype=torch.float16)

model.save_pretrained("zephyr_gptq_2")
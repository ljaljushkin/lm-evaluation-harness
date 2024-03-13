import argparse
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from peft import LoftQConfig, LoraConfig, get_peft_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_id", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-l", "--layers",  nargs='+', default=['down_proj', 'o_proj'])
    parser.add_argument("-r", "--rank", type=int, default=8)
    parser.add_argument("--fp32", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()
    model_id = args.model_id
    output_dir = args.output_dir

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not args.fp32:
        lora_config = LoraConfig(
            target_modules=args.layers,
            r=args.rank,
            init_lora_weights=False
        )
        model.add_adapter(lora_config)
        model.enable_adapters()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.config.to_json_file(output_dir + '/config.json')

    text = "Hello"
    inputs = tokenizer(text, return_tensors="pt")
    if not args.fp32:
        output_enabled = model.generate(**inputs, max_new_tokens=10)
        print('ENABLED adapters: ', tokenizer.decode(output_enabled[0], skip_special_tokens=True))
        model.disable_adapters()

    output_disabled = model.generate(**inputs, max_new_tokens=10)
    print('DISABLED adapters: ', tokenizer.decode(output_disabled[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
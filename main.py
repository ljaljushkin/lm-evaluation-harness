import argparse
import json
import logging
import os

from lm_eval import tasks, evaluator, utils

import argparse
from datetime import datetime
import json
import logging
import fnmatch
from pathlib import Path
from time import time
import random

import torch
from transformers import AutoModelForCausalLM
from correct_ir import correct_attention_mask_names

from lm_eval import tasks, evaluator
# from visualization import parse_results
from openvino.runtime import serialize
from openvino.tools.mo import convert_model

from nncf import compress_weights

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    model_name = Path(args.model_args).name
    random.seed(42)
    date = datetime.now().strftime("%b%d_%H-%M-%S")
    cache_dir = Path('cache') / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    encoded_name = 'int4__first_last_kqv_ffn2_int8'

    log_dir = Path('runs') / model_name / f'{encoded_name}_{date}'
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / 'args.json').open('w') as f:
        json.dump(vars(args), f, indent=4)
    model_args = args.model_args
    ir_cache_dir = cache_dir / encoded_name
    ir_path = ir_cache_dir / 'openvino_model.xml'
    time_dict = {}
    if not ir_path.exists():
        ir_cache_dir.mkdir(exist_ok=True)
        model_id = args.model_args.split('pretrained=')[1].split(',')[0]
        model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=False, trust_remote_code=True)#, torch_dtype=torch.bfloat16)
        print(model)
        model.config.save_pretrained(ir_cache_dir)

        if encoded_name.startswith('int'):
            start_time = time()
            print(f'started weights compression')
            model = compress_weights(model, use_fake_quantize=False)
            nncf_time = time() - start_time
            time_dict['nncf'] = nncf_time
            print(f'weights compression took {nncf_time} seconds')

        start_time = time()
        print(f'started mo convert')
        example_input = {
            "input_ids": torch.ones([1,2],dtype=torch.long),
            "attention_mask": torch.ones([1,2], dtype=torch.long),
        }
        ov_model = convert_model(model, example_input=example_input)
        # apply_moc_transformations(ov_model)
        # apply_fused_names_cleanup(ov_model)
        mo_time = time() - start_time
        time_dict['mo'] = mo_time
        print(f'mo convert took {mo_time} seconds')

        serialize(ov_model, ir_path)

        correct_attention_mask_names(ir_path)

    model_args = f'pretrained={ir_cache_dir.resolve()}'

    start_time = time()
    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
        tokenizer=args.tokenizer,
    )
    eval_time = time() - start_time
    time_dict['eval'] = eval_time
    print(f'eval took {eval_time} seconds')
    results['time'] = time_dict
    with (log_dir / 'results.json').open('w') as f:
        json.dump(results, f, indent=2)
    print(evaluator.make_table(results))



if __name__ == "__main__":
    main()

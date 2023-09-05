import argparse
import json
import logging
import os

from lm_eval import tasks, evaluator, utils
import shutil
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
import torch
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
# from memory_profiler import memory_usage
from optimum.intel import OVModelForCausalLM

from optimum.intel.openvino import OVConfig, OVQuantizer
import datasets
import nncf

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", required=True)
    # parser.add_argument(
    #     "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    # )
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=100)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true", default=True)
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--delete_ir_cache", action="store_true", default=False)
    parser.add_argument("--do_eval", action="store_true", default=False)

    return parser.parse_args()

def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    # if args.tasks is None:
    #     task_names = tasks.ALL_TASKS
    # else:
    #     task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    # print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    use_pkv = True
    experiments = {
        'databricks/dolly-v2-3b': dict(group_size=64, mode='nf4', delete_ir_cache=True, limit=100),
        # 'databricks/dolly-v2-3b': dict(group_size=64, mode='uni', is_mixed=False),
        # 'databricks/dolly-v2-3b': dict(group_size=64, mode='pq', is_mixed=False)
    }

    for model_id, config in experiments.items():
        model_name = Path(model_id).name
        random.seed(42)
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        cache_dir = Path('cache') / model_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        group_size = config['group_size']
        mode = config['mode']
        is_mixed = config.get('is_mixed', False)
        delete_ir_cache = config.get('delete_ir_cache', False)
        limit = config.get('limit')
        do_eval = config.get('do_eval', True)
        group_str = f'_g{group_size}' if group_size >= 2 else ''
        model_args = f'pretrained={model_id}'
        mixed_str = '_mixed' if is_mixed else ''
        encoded_name = f'{mode}{group_str}{mixed_str}'

        log_dir = Path('runs') / model_name / f'{encoded_name}_{date}'
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / 'args.json').open('w') as f:
            json.dump(vars(args), f, indent=4)

        ir_cache_dir = cache_dir / encoded_name
        ir_path = ir_cache_dir / 'openvino_model.xml'
        if delete_ir_cache:# and ir_path.exists():
            # TODO: remove all except folder with results.json
            shutil.rmtree(ir_cache_dir)
            # print('remove IRs:')
            # for file_to_remove in ir_cache_dir.glob('openvino_model.*'):
            #     print(file_to_remove)
            #     Path.unlink(file_to_remove)
        ir_cache_dir.mkdir(exist_ok=True)
        os.symlink(ir_cache_dir.resolve(), log_dir.resolve() / ir_cache_dir.name)
        os.symlink(log_dir.resolve(), ir_cache_dir.resolve() / log_dir.name)
        time_dict = {}
        if not ir_path.exists():
            if 'fp32' not in encoded_name:
                print(f'started weights compression')
                start_time = time()
                quantization_config = {
                    "algorithm": "quantization"
                }
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, use_cache=use_pkv, trust_remote_code=True,
                    # TODO: aidova tip to avoid issue with model.onnx and probably with compilation
                    # torchscript=True,
                    use_auth_token=True
                )
                print(model)
                tokenizer = AutoTokenizer.from_pretrained(model_id)

                config = OVConfig(compression=quantization_config)
                config.target_device = "TRIAL"
                tokenizer.pad_token = tokenizer.eos_token

                quantizer = OVQuantizer.from_pretrained(model)

                if hasattr(model, "transformer") and hasattr(model.transformer, "wte") and type(model.transformer.wte) != torch.nn.Embedding:
                    from nncf.torch import register_module
                    register_module(ignored_algorithms=[], target_weight_dim_for_compression=1)(type(model.transformer.wte))

                quantizer.quantize(save_directory=ir_cache_dir, weights_only=True, group_size=64, mode='nf4', is_mixed=False)

                nncf_time = time() - start_time
                time_dict['nncf'] = nncf_time
                print(f'weights compression took {nncf_time} seconds')
            else:
                ov_model = OVModelForCausalLM.from_pretrained(model_id, use_cache=use_pkv, trust_remote_code=True, from_transformers=True)
                ov_model.save_pretrained(ir_cache_dir)

        model_args = f'pretrained={ir_cache_dir.resolve()}'

        if do_eval:
            start_time = time()
            results = evaluator.simple_evaluate(
                model='optimum-causal',
                model_args=model_args,
                tasks=['lambada_openai'],
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                max_batch_size=args.max_batch_size,
                device=args.device,
                no_cache=args.no_cache,
                limit=limit,
                description_dict=description_dict,
                decontamination_ngrams_path=args.decontamination_ngrams_path,
                check_integrity=args.check_integrity,
                write_out=args.write_out,
                output_base_path=args.output_base_path,
                tokenizer=model_id,
            )
            eval_time = time() - start_time
            time_dict['eval'] = eval_time
            print(f'eval took {eval_time} seconds')
            results['time'] = time_dict
            with (log_dir / 'results.json').open('w') as f:
                json.dump(results, f, indent=2)
            print(evaluator.make_table(results))

        model_cache_dir = ir_cache_dir / 'model_cache'
        if model_cache_dir.exists():
            Path.unlink(model_cache_dir)



if __name__ == "__main__":
    main()

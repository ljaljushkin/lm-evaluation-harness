
from pathlib import Path
import shutil


SHARE_DIR = Path('/mnt/icv_externalN/users/nlyalyus/models')
CACHE_DIR = Path('./cache')

MODEL_IDS = [
    # 'facebook/opt-125m',
    'databricks/dolly-v2-3b',
    'openlm-research/open_llama_3b',
    'facebook/opt-6.7b',
    'bigscience/bloom-7b1',
    'togethercomputer/RedPajama-INCITE-7B-Instruct',
    'databricks/dolly-v2-12b',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-13b-chat-hf',
    # 'chatglm2-6b',
]

EXP_NAMES = [
    'nf4_ov_g64',
    'nf4_ov_g128',
    'nf4_ov',
]


for model_id in MODEL_IDS:
    for exp_name in EXP_NAMES:
        try:
            model_name = Path(model_id).name
            src_dir = CACHE_DIR / model_name / exp_name
            dst_dir = SHARE_DIR / model_name / exp_name
            if not dst_dir.exists():
                dst_dir.mkdir(exist_ok=True)
            shutil.copyfile(src_dir / 'config.json', dst_dir / 'config.json')
            for src_path in src_dir.glob('openvino_model.*'):
                dst_path = dst_dir / src_path.name
                print(f'copying {src_path} to {dst_path}')
                shutil.copyfile(src_path, dst_path)
            res_path = dst_dir / 'results.json'
            if res_path.exists():
                Path.unlink(res_path)
        except Exception as error:
            print(f"copy of {model_id} {exp_name} failed with message: {error}")
            continue

from pathlib import Path
from nncf import compress_weights
import traceback
from openvino import Core
import json
import nncf
core = Core()

llm_model_dir = Path('/mnt/cifs/ov-share-05/chunk-01/openvino_models/models')
num_ok = 0
num_all = 0
# for fp32_dirs in llm_model_dir.rglob('*FP32*'):
#     for model_path in fp32_dirs.rglob('*openvino_model.xml'):
#         print('Found: ', model_path.resolve())
#         num_all += 1
try:
    model_path = '/mnt/cifs/ov-share-05/chunk-01/openvino_models/models/stable-diffusion-v1-5/onnx/dldt/FP32/text_encoder/openvino_model.xml'
    fp32_model = core.read_model(model=model_path)
    # json_path = model_path.parent / 'config.json'
    # with json_path.open() as f:
    #     j = json.load(f)
    #     model_list.append(j['_name_or_path'])
    compress_weights(fp32_model, mode=nncf.CompressWeightsMode.INT4_ASYM)
    num_ok += 1
except Exception as error:
    print(traceback.print_exc())
    print(f"Compression failed with error: {error}")
print(num_ok, num_all)
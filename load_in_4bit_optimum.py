import nncf
import os
from pathlib import Path
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

MODEL_ID = "databricks/dolly-v2-3b"
model_name = MODEL_ID.split('/')[1]
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = OVModelForCausalLM.from_pretrained(
    MODEL_ID,
    export=True,
    load_in_4bit=True,
)

CACHE_DIR = Path(os.readlink('cache'))
model_folder = CACHE_DIR / model_name / 'tmp'
model_folder.mkdir(exist_ok=True, parents=True)
print('saving model to: ', str(model_folder.resolve()))
model.save_pretrained(model_folder)
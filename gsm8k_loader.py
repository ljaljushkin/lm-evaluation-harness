import datasets
from transformers import AutoTokenizer

seqlen = 512
model_id = 'stabilityai/stablelm-2-zephyr-1_6b'
nsamples = 64


tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = datasets.load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
print(next(iter(dataset)))
traindata = dataset.take(nsamples)
trainloader = []
for i in range(nsamples):
    trainenc = tokenizer(traindata[i]["question"], return_tensors="pt")
    # print(trainenc)
    if trainenc.input_ids.shape[1] > seqlen:
        print(f'more than {seqlen}: {trainenc.input_ids.shape[1]}')
        break
    trainloader.append(trainenc)



#         # trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
#         def preprocess_fn(example):
#             return {"prompt": example["caption"]}
#         dataset = dataset.map(lambda x: preprocess_fn(x), remove_columns=dataset.column_names)
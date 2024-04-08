from datasets import load_dataset

dataset = load_dataset(
    path='EleutherAI/wikitext_document_level',
    name='wikitext-2-raw-v1',
    data_dir=None,
    cache_dir=None,
    download_mode=None,
)

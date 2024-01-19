"""
The LAMBADA dataset: Word prediction requiring a broad discourse context∗
https://arxiv.org/pdf/1606.06031.pdf

LAMBADA is a dataset to evaluate the capabilities of computational models for text
understanding by means of a word prediction task. LAMBADA is a collection of narrative
passages sharing the characteristic that human subjects are able to guess their last
word if they are exposed to the whole passage, but not if they only see the last
sentence preceding the target word. To succeed on LAMBADA, computational models
cannot simply rely on local context, but must be able to keep track of information
in the broader discourse.

Homepage: https://zenodo.org/record/2630551#.X4Xzn5NKjUI
"""
from lm_eval.base import Task, rf
from lm_eval.metrics import mean, perplexity
import re

_CITATION = """
@misc{
    author={Paperno, Denis and Kruszewski, Germán and Lazaridou, Angeliki and Pham, Quan Ngoc and Bernardi, Raffaella and Pezzelle, Sandro and Baroni, Marco and Boleda, Gemma and Fernández, Raquel},
    title={The LAMBADA dataset},
    DOI={10.5281/zenodo.2630551},
    publisher={Zenodo},
    year={2016},
    month={Aug}
}
"""


class LambadaBase(Task):
    VERSION = None

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return doc["text"].rsplit(" ", 1)[0]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_target(self, doc):
        return " " + doc["text"].rsplit(" ", 1)[1]

    def construct_requests(self, doc, ctx):
        ll, is_greedy = rf.loglikelihood(ctx, self.doc_to_target(doc))

        return ll, is_greedy

    def process_results(self, doc, results):
        ll, is_greedy = results

        return {"ppl": ll, "acc": int(is_greedy)}

    def aggregation(self):
        return {"ppl": perplexity, "acc": mean}

    def higher_is_better(self):
        return {"ppl": False, "acc": True}


class LambadaStandard(LambadaBase):
    """The LAMBADA task using the standard original LAMBADA dataset."""

    VERSION = 0
    DATASET_PATH = "lambada"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True


class LambadaOpenAI(LambadaBase):
    """The LAMBADA task using the LAMBADA OpenAI dataset, a modified version of the
    original LAMBADA dataset created by OpenAI for evaluating their GPT-2 model.

    Reference: https://github.com/openai/gpt-2/issues/131#issuecomment-497136199
    """

    VERSION = 0
    DATASET_PATH = "EleutherAI/lambada_openai"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

import transformers

class WikitextZhYue(LambadaBase):
    """The LAMBADA task using the LAMBADA OpenAI dataset, a modified version of the
    original LAMBADA dataset created by OpenAI for evaluating their GPT-2 model.

    Reference: https://github.com/openai/gpt-2/issues/131#issuecomment-497136199
    """

    VERSION = 0
    DATASET_PATH = "indiejoseph/wikitext-zh-yue"
    # DATASET_PATH = "clue"
    # DATASET_NAME = "iflytek"

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        super().__init__(data_dir, cache_dir, download_mode)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            # 'Qwen/Qwen-7B-Chat',
            'THUDM/chatglm3-6b',
            trust_remote_code=True,
            # pad_token='<|extra_0|>',
            # eos_token='<|endoftext|>',
            # padding_side='left'
        )

    def _remove_noise(self, string):
        noisy_symbols = ("。", "；", "~", ".", "，", "；")
        while string.endswith(noisy_symbols):
            string = string[:-1]
        return string

    def test_docs(self):
        if self.has_test_docs():
            test_dataset = self.dataset["test"]
            test_dataset = test_dataset.filter(
                lambda example: len(example['text']) > 107
            )
            return test_dataset#.select(range(5))

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def _get_last_token_length(self, sentence):
        list_of_tokens = self.tokenizer.batch_decode(self.tokenizer(sentence).input_ids)
        last_token = list_of_tokens[-1]
        last_token_length = len(last_token)
        return last_token_length

    def doc_to_target(self, doc):
        # sentence = self._remove_noise(doc['sentence'])
        sentence = self._remove_noise(doc['text'])
        last_token_length = self._get_last_token_length(sentence)
        result = sentence[-last_token_length:]
        # print(result)
        return result

    def doc_to_text(self, doc):
        # sentence = self._remove_noise(doc['sentence'])
        sentence = self._remove_noise(doc['text'])
        last_token_length = self._get_last_token_length(sentence)
        result = sentence[:-last_token_length]
        # print(result[-5:])
        return result

    def doc_to_decontamination_query(self, doc):
        # return doc["sentence"]
        return doc["text"]

class CLUE(LambadaBase):
    """The LAMBADA task using the LAMBADA OpenAI dataset, a modified version of the
    original LAMBADA dataset created by OpenAI for evaluating their GPT-2 model.

    Reference: https://github.com/openai/gpt-2/issues/131#issuecomment-497136199
    """

    VERSION = 0
    DATASET_PATH = "clue"
    DATASET_NAME = "iflytek"

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        super().__init__(data_dir, cache_dir, download_mode)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            # 'Qwen/Qwen-7B-Chat',
            'THUDM/chatglm3-6b',
            trust_remote_code=True,
            # pad_token='<|extra_0|>',
            # eos_token='<|endoftext|>',
            # padding_side='left'
        )

    def _remove_noise(self, string):
        noisy_symbols = ("。", "；", "~", ".", "，", "；")
        while string.endswith(noisy_symbols):
            string = string[:-1]
        return string

    def test_docs(self):
        if self.has_test_docs():
            test_dataset = self.dataset["test"]
            # test_dataset = test_dataset.filter(
            #     lambda example: len(example['text']) > 107
            # )
            return test_dataset#.select(range(5))

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def _get_last_token_length(self, sentence):
        list_of_tokens = self.tokenizer.batch_decode(self.tokenizer(sentence).input_ids)
        last_token = list_of_tokens[-1]
        last_token_length = len(last_token)
        return last_token_length

    def doc_to_target(self, doc):
        sentence = self._remove_noise(doc['sentence'])
        last_token_length = self._get_last_token_length(sentence)
        result = sentence[-last_token_length:]
        return result

    def doc_to_text(self, doc):
        sentence = self._remove_noise(doc['sentence'])
        last_token_length = self._get_last_token_length(sentence)
        result = sentence[:-last_token_length]
        return result

    def doc_to_decontamination_query(self, doc):
        return doc["sentence"]


class C4_ZH(LambadaBase):
    """The LAMBADA task using the LAMBADA OpenAI dataset, a modified version of the
    original LAMBADA dataset created by OpenAI for evaluating their GPT-2 model.

    Reference: https://github.com/openai/gpt-2/issues/131#issuecomment-497136199
    """

    VERSION = 0
    DATASET_PATH = "allenai/c4"
    DATASET_NAME = "zh"

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        super().__init__(data_dir, cache_dir, download_mode)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            # 'Qwen/Qwen-7B-Chat',
            'THUDM/chatglm3-6b',
            trust_remote_code=True,
            # pad_token='<|extra_0|>',
            # eos_token='<|endoftext|>',
            # padding_side='left'
        )

    def _remove_noise(self, string):
        noisy_symbols = ("。", "；", "~", ".", "，", "；")
        while string.endswith(noisy_symbols):
            string = string[:-1]
        return string

    def validation_docs(self):
        test_dataset = self.dataset["validation"]
        # test_dataset = test_dataset.filter(
        #     lambda example: len(example['text']) > 107
        # )
        return test_dataset#.select(range(5))

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def _get_last_token_length(self, sentence):
        list_of_tokens = self.tokenizer.batch_decode(self.tokenizer(sentence).input_ids)
        last_token = list_of_tokens[-1]
        last_token_length = len(last_token)
        return last_token_length

    def doc_to_target(self, doc):
        sentence = self._remove_noise(doc['sentence'])
        last_token_length = self._get_last_token_length(sentence)
        result = sentence[-last_token_length:]
        return result

    def doc_to_text(self, doc):
        sentence = self._remove_noise(doc['sentence'])
        last_token_length = self._get_last_token_length(sentence)
        result = sentence[:-last_token_length]
        return result

    def doc_to_decontamination_query(self, doc):
        return doc["sentence"]


class Alpaca_ZH(LambadaBase):
    """The LAMBADA task using the LAMBADA OpenAI dataset, a modified version of the
    original LAMBADA dataset created by OpenAI for evaluating their GPT-2 model.

    Reference: https://github.com/openai/gpt-2/issues/131#issuecomment-497136199
    """

    VERSION = 0
    DATASET_PATH = "shibing624/alpaca-zh"

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        super().__init__(data_dir, cache_dir, download_mode)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            # 'Qwen/Qwen-7B-Chat',
            'THUDM/chatglm3-6b',
            trust_remote_code=True,
            # pad_token='<|extra_0|>',
            # eos_token='<|endoftext|>',
            # padding_side='left'
        )

    def _remove_noise(self, string):
        noisy_symbols = ("。", "；", "~", ".", "，", "；", ".", "?")
        while string.endswith(noisy_symbols):
            string = string[:-1]
        return string

    def test_docs(self):
        dataset = self.dataset["train"]
        dataset = dataset.filter(
            lambda example: 100 < len(example['output']) < 400 and not re.search('[a-zA-Z]', example['output'])
        )
        return dataset#.select(range(5))

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def _get_last_token_length(self, sentence):
        list_of_tokens = self.tokenizer.batch_decode(self.tokenizer(sentence).input_ids)
        last_token = list_of_tokens[-1]
        last_token_length = len(last_token)
        return last_token_length

    def doc_to_target(self, doc):
        sentence = self._remove_noise(doc['output'])
        last_token_length = self._get_last_token_length(sentence)
        result = sentence[-last_token_length:]
        return result

    def doc_to_text(self, doc):
        sentence = self._remove_noise(doc['output'])
        last_token_length = self._get_last_token_length(sentence)
        result = sentence[:-last_token_length]
        return result

    def doc_to_decontamination_query(self, doc):
        return doc["output"]



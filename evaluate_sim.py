import os
from pathlib import Path
from typing import List
import traceback
from nncf import compress_weights
from tqdm import trange
import json
import shutil
import random
import copy
from optimum.intel.openvino import (
    OVModelForCausalLM,
    OVQwenModel
)
from transformers import (
    AutoTokenizer,
    AutoConfig
)
from contextlib import redirect_stdout, redirect_stderr
from nncf.parameters import CompressWeightsMode
from whowhatbench import Evaluator

from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
from optimum.exporters import TasksManager
from openvino import Core
core = Core()
from datasets import load_dataset

# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
# )

# from transformers import GenerationConfig, StoppingCriteriaList
# from transformers.generation.logits_process import LogitsProcessorList, LogitsProcessor
# from transformers.generation.utils import GenerateOutput
# from openvino.runtime import Model, Core, Tensor, Type


# from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
# from optimum.exporters import TasksManager

# import time
# import inspect
# from pathlib import Path
# from typing import Optional, Union, Dict, List, Tuple, Callable, Iterable, Any
# from tempfile import TemporaryDirectory
# import PIL
# import numpy as np
# import torch
# from optimum.intel.openvino.utils import ONNX_WEIGHTS_NAME, OV_XML_FILE_NAME
# from openvino.runtime import Model, Core, Tensor, Type
# from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
# from transformers import PretrainedConfig
# from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
# from transformers import GenerationConfig, StoppingCriteriaList
# from transformers.generation.logits_process import LogitsProcessorList, LogitsProcessor
# from transformers.generation.utils import GenerateOutput


# class StopWordsLogitsProcessor(LogitsProcessor):
#     '''
#     :class:`transformers.LogitsProcessor` that enforces that when specified sequences appear, stop geration.

#     Args:
#         stop_words_ids (:obj:`List[List[int]]`):
#             List of list of token ids of stop ids. In order to get the tokens of the words
#             that should not appear in the generated text, use :obj:`tokenizer(bad_word,
#             add_prefix_space=True).input_ids`.
#         eos_token_id (:obj:`int`):
#             The id of the `end-of-sequence` token.
#     '''

#     def __init__(self, stop_words_ids: Iterable[Iterable[int]], eos_token_id: int):

#         if not isinstance(stop_words_ids, List) or len(stop_words_ids) == 0:
#             raise ValueError(
#                 f'`stop_words_ids` has to be a non-emtpy list, but is {stop_words_ids}.'
#             )
#         if any(not isinstance(bad_word_ids, list) for bad_word_ids in stop_words_ids):
#             raise ValueError(
#                 f'`stop_words_ids` has to be a list of lists, but is {stop_words_ids}.'
#             )
#         if any(
#             any(
#                 (not isinstance(token_id, (int, np.integer)) or token_id < 0)
#                 for token_id in stop_word_ids
#             )
#             for stop_word_ids in stop_words_ids
#         ):
#             raise ValueError(
#                 f'Each list in `stop_words_ids` has to be a list of positive integers, but is {stop_words_ids}.'
#             )

#         self.stop_words_ids = list(
#             filter(
#                 lambda bad_token_seq: bad_token_seq != [eos_token_id], stop_words_ids
#             )
#         )
#         self.eos_token_id = eos_token_id
#         for stop_token_seq in self.stop_words_ids:
#             assert (
#                 len(stop_token_seq) > 0
#             ), 'Stop words token sequences {} cannot have an empty list'.format(
#                 stop_words_ids
#             )

#     def __call__(
#         self, input_ids: torch.LongTensor, scores: torch.FloatTensor
#     ) -> torch.FloatTensor:
#         stopped_samples = self._calc_stopped_samples(input_ids)
#         for i, should_stop in enumerate(stopped_samples):
#             if should_stop:
#                 scores[i, self.eos_token_id] = float(2**15)
#         return scores

#     def _tokens_match(self, prev_tokens: torch.LongTensor, tokens: List[int]) -> bool:
#         if len(tokens) == 0:
#             # if bad word tokens is just one token always ban it
#             return True
#         elif len(tokens) > len(prev_tokens):
#             # if bad word tokens are longer then prev input_ids they can't be equal
#             return False
#         elif prev_tokens[-len(tokens) :].tolist() == tokens:
#             # if tokens match
#             return True
#         else:
#             return False

#     def _calc_stopped_samples(self, prev_input_ids: Iterable[int]) -> Iterable[int]:
#         stopped_samples = []
#         for prev_input_ids_slice in prev_input_ids:
#             match = False
#             for stop_token_seq in self.stop_words_ids:
#                 if self._tokens_match(prev_input_ids_slice, stop_token_seq):
#                     # if tokens do not match continue
#                     match = True
#                     break
#             stopped_samples.append(match)

#         return stopped_samples

# class OVQwenModel(OVModelForCausalLM):
#     def _reshape(
#         self,
#         model: Model,
#         batch_size: int,
#         sequence_length: int,
#         height: int = None,
#         width: int = None,
#     ):
#         shapes = {}
#         for inputs in model.inputs:
#             shapes[inputs] = inputs.get_partial_shape()
#             shapes[inputs][0] = -1
#             shapes[inputs][1] = -1
#         model.reshape(shapes)
#         return model

#     @classmethod
#     def _from_pretrained(
#         cls,
#         model_id: Union[str, Path],
#         config: PretrainedConfig,
#         use_auth_token: Optional[Union[bool, str, None]] = None,
#         revision: Optional[Union[str, None]] = None,
#         force_download: bool = False,
#         cache_dir: Optional[str] = None,
#         file_name: Optional[str] = None,
#         subfolder: str = '',
#         from_onnx: bool = False,
#         local_files_only: bool = False,
#         load_in_8bit: bool = False,
#         **kwargs,
#     ):
#         model_path = Path(model_id)
#         default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
#         file_name = file_name or default_file_name

#         model_cache_path = cls._cached_file(
#             model_path=model_path,
#             use_auth_token=use_auth_token,
#             revision=revision,
#             force_download=force_download,
#             cache_dir=cache_dir,
#             file_name=file_name,
#             subfolder=subfolder,
#             local_files_only=local_files_only,
#         )

#         model = cls.load_model(model_cache_path, load_in_8bit=load_in_8bit)
#         init_cls = OVQwenModel

#         return init_cls(model=model, config=config, model_save_dir=model_cache_path.parent, **kwargs)

#     def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
#         past_key_values = past_key_values or kwargs.get('past', None)

#         # `past_key_values` may be in the stardard format (e.g. in contrastive search), converts to bloom's format if needed
#         if past_key_values is not None and self.config.model_type == 'bloom':
#             if past_key_values[0][0].shape[0] == input_ids.shape[0]:
#                 past_key_values = self._convert_to_bloom_cache(past_key_values)

#         attention_mask = kwargs.get('attention_mask', None)
#         position_ids = kwargs.get('position_ids', None)
#         if attention_mask is not None and position_ids is None:
#             # create position_ids on the fly for batch generation
#             position_ids = attention_mask.long().cumsum(-1) - 1
#             position_ids.masked_fill_(attention_mask == 0, 1)
#         if past_key_values:
#             position_ids = position_ids[:, -1].unsqueeze(-1)
#         return {
#             'input_ids': input_ids,
#             'past_key_values': past_key_values,
#             'use_cache': self.use_cache,
#             'position_ids': position_ids,
#             'attention_mask': attention_mask,
#             'token_type_ids': None,
#         }

#     def _update_model_kwargs_for_generation(
#         self,
#         outputs: 'ModelOutput',
#         model_kwargs: Dict[str, 'Any'],
#         is_encoder_decoder: bool = False,
#         standardize_cache_format: bool = False,
#     ) -> Dict[str, 'Any']:
#         # update past_key_values
#         model_kwargs['past_key_values'] = self._extract_past_from_model_output(
#             outputs, standardize_cache_format=standardize_cache_format
#         )

#         # update attention mask
#         if 'attention_mask' in model_kwargs:
#             attention_mask = model_kwargs['attention_mask']
#             model_kwargs['attention_mask'] = torch.cat(
#                 [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
#             )

#         # update position ids
#         if 'position_ids' in model_kwargs:
#             position_ids = model_kwargs['position_ids']
#             new_position_id = position_ids[..., -1:].clone()
#             new_position_id += 1
#             model_kwargs['position_ids'] = torch.cat([position_ids, new_position_id], dim=-1)

#         model_kwargs['is_first_forward'] = False
#         return model_kwargs


#     def generate(
#         self,
#         inputs: Optional[torch.Tensor] = None,
#         generation_config: Optional[GenerationConfig] = None,
#         logits_processor: Optional[LogitsProcessorList] = None,
#         stopping_criteria: Optional[StoppingCriteriaList] = None,
#         prefix_allowed_tokens_fn: Optional[
#             Callable[[int, torch.Tensor], List[int]]
#         ] = None,
#         synced_gpus: Optional[bool] = None,
#         #assistant_model: Optional['PreTrainedModel'] = None,
#         #streamer: Optional['BaseStreamer'] = None,
#         **kwargs,
#     ) -> Union[GenerateOutput, torch.LongTensor]:
#         generation_config = generation_config if generation_config is not None else self.generation_config

#         # Process stop_words_ids.
#         stop_words_ids = kwargs.pop('stop_words_ids', [[151643]])
#         if stop_words_ids is None and generation_config is not None:
#             stop_words_ids = getattr(generation_config, 'stop_words_ids', None)
#         if stop_words_ids is None:
#             stop_words_ids = getattr(generation_config, 'stop_words_ids', None)

#         if stop_words_ids is not None:
#             stop_words_logits_processor = StopWordsLogitsProcessor(
#                 stop_words_ids=stop_words_ids,
#                 eos_token_id=generation_config.eos_token_id,
#             )
#             if logits_processor is None:
#                 logits_processor = LogitsProcessorList([stop_words_logits_processor])
#             else:
#                 logits_processor.append(stop_words_logits_processor)

#         return super().generate(
#             inputs,
#             generation_config=generation_config,
#             logits_processor=logits_processor,
#             stopping_criteria=stopping_criteria,
#             prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
#             synced_gpus=synced_gpus,
#             **kwargs,
#         )


TasksManager._SUPPORTED_MODEL_TYPE["stablelm-epoch"] = TasksManager._SUPPORTED_MODEL_TYPE["llama"]
NormalizedConfigManager._conf["stablelm-epoch"] = NormalizedTextConfig.with_args(
    num_layers="num_hidden_layers",
    num_attention_heads="num_attention_heads",
)
NormalizedConfigManager._conf['qwen'] = NormalizedTextConfig.with_args(
    num_layers='num_hidden_layers', num_attention_heads='num_attention_heads', hidden_size='hidden_size'
)

MODEL_IDS = [
    # 'stable-zephyr-3b-dpo',
    # 'llama-2-7b-chat-hf',
    # 'stablelm-3b-4e1t',
    # 'zephyr-7b-beta',
    # 'qwen-7b-chat',
    'facebook/opt-125m',
]
for MODEL_ID in MODEL_IDS:
    model_name = Path(MODEL_ID).name
    cache_dir = Path(os.readlink('/home/devuser/nlyalyus/projects/lm-evaluation-harness/cache'))
    ROOT_DIR = cache_dir / model_name
    gold_folder = ROOT_DIR / "fp16"
    gold_ir_path = gold_folder / "openvino_model.xml"
    gold_csv = gold_folder / 'gold_all.csv'
    config_gold = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    # config_gold = AutoConfig.from_pretrained('Qwen/Qwen-7B-Chat', trust_remote_code=True)
    # tokenizer_gold = AutoTokenizer.from_pretrained('Qwen/Qwen-7B-Chat', trust_remote_code=True )
    tokenizer_gold = AutoTokenizer.from_pretrained(MODEL_ID)
    # tokenizer_gold = AutoTokenizer.from_pretrained('Qwen/Qwen-7B-Chat')
    tokenizer_gold.save_pretrained(gold_folder)
    print('gold path:', gold_csv.resolve())
    if gold_csv.exists():
        evaluator = Evaluator(tokenizer=tokenizer_gold, gt_data=gold_csv, test_data=str(gold_csv))
        # evaluator = Evaluator(
        #     tokenizer=tokenizer_gold, gt_data=gold_csv, test_data=str(gold_csv),
        #     similarity_model_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        # )
    else:
        # model_gold = OVQwenModel.from_pretrained(gold_folder, config=config_gold, trust_remote_code=True, use_cache=True)
        # model_gold._reshape(model_gold.model, -1, -1)
        model_gold = OVModelForCausalLM.from_pretrained(gold_folder, config=config_gold, trust_remote_code=True, use_cache=True)
        evaluator = Evaluator(base_model=model_gold, tokenizer=tokenizer_gold)
        # dataset = load_dataset('ceval/ceval-exam', 'high_school_geography', split='test')
        # prompts = list(dataset["question"])[:24]
        # evaluator = Evaluator(base_model=model_gold, tokenizer=tokenizer_gold, test_data=prompts,
        #                       similarity_model_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        evaluator.dump_gt(str(gold_csv))

    EXP_NAMES = [
        # 'int4_sym_g64_r80_greedy0_anti',
        # 'int4_sym_g64_r80_greedy0',
        # 'int4_sym_g64_r80_greedy1',

        # 'int4_sym_g64_r80',
        # 'int4_sym_g64_r80_hawq_in',
        # 'int4_sym_g64_r80_max_var',
        # 'int4_sym_g64_r80_mean_var',
        # 'int4_sym_g64_r80_mean_max',

        'int8',
        'int4_asym_g128_r80',
        # 'int4_sym_g128_r80_hawq_in',
        'int4_asym_g128_r80_max_var',
        # 'int4_sym_g128_r80_mean_var',
        # 'int4_sym_g128_r80_mean_max',
        # 'fp16'

        # 'int4_sym_g64_r80_criteria_OUT2',
        # 'int4_sym_g64_r80_baseline',
        # 'int4_sym_g64_r80_mean_var',
        # 'int4_sym_g64_r80_max_var',
        # 'int4_sym_g64_r80_mean_max',
    ]

    for exp_name in EXP_NAMES:
        cmp_ir_folder = ROOT_DIR / exp_name
        cmp_ir_path = cmp_ir_folder / "openvino_model.xml"
        # cmp_model = core.read_model(model=cmp_ir_path)
        cmp_model = OVModelForCausalLM.from_pretrained(cmp_ir_folder, config=config_gold, trust_remote_code=True, use_cache=True)
        # cmp_model = OVQwenModel(cmp_model, config=config_gold, trust_remote_code=True, use_cache=True, model_save_dir=cmp_ir_folder)
        # cmp_model = OVQwenModel.from_pretrained(cmp_ir_folder, config=config_gold, device_map="auto", trust_remote_code=True)
        # cmp_model._reshape(cmp_model.model, -1, -1)
        all_metrics_per_question, all_metrics = evaluator.score(cmp_model)
        print(all_metrics)
        similarity = float(all_metrics['similarity'].iloc[0])
        sdt_norm = float(all_metrics['SDT norm'].iloc[0])
        score = (similarity + 1 - sdt_norm) / 2
        # print(all_metrics_per_question)
        print('final score=', score)
        all_metrics['weighted'] = [score]
        model_cache_dir = cmp_ir_folder / 'model_cache'
        if model_cache_dir.exists():
            shutil.rmtree(model_cache_dir)
        all_metrics.to_csv(cmp_ir_folder / 'eval.csv')
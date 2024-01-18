import torch
import transformers
import optimum
from optimum.intel.openvino import OVModelForCausalLM

from typing import Optional
from lm_eval.base import BaseLM
# from optimum.intel.openvino import OVMistralModel
# from optimum.intel.openvino import OVQwenModel
# from optimum.intel.openvino import OVChatGLM2Model
from transformers import AutoConfig

class OptimumIntelAutoCausalLM(BaseLM):
    def __init__(
        self,
        device="cpu",
        pretrained="gpt2",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = True,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int,str))

        self._device = "cpu"

        revision = revision + ("/" + subfolder if subfolder is not None else "")

        # from optimum.intel.openvino import OVChatGLM2Model
        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)

        # self.model = OVMistralModel.from_pretrained(
        # self.model = OVChatGLM2Model.from_pretrained(
        # self.model = OVQwenModel.from_pretrained(
        # NOTE: StableLM support
        self.model = OVModelForCausalLM.from_pretrained(
            pretrained,
            config=config,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_cache=True,
            # from_transformers=True
        )
        # self.model = {}

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            # revision=revision,
            trust_remote_code=trust_remote_code,
            # pad_token='<|extra_0|>',
            # eos_token='<|endoftext|>',
            # padding_side='left'
        )

        self.vocab_size = self.tokenizer.vocab_size

        # setup for automatic batch size detection
        if batch_size == 'auto':
            self.batch_size_per_gpu = batch_size
        else:
            self.batch_size_per_gpu = int(batch_size)

        self._DEFAULT_MAX_LENGTH = 512


    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id


    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig, T5Config)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        #with torch.no_grad():
        attention_mask = inps.clone()
        attention_mask[:] = 1.0
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        # if past_key_values:
        #     position_ids = position_ids[:, -1].unsqueeze(-1)
        # position_ids = torch.range(0, inps.shape[1] + 1, dtype=inps.dtype).repeat(1, 1)
        return self.model(inps, attention_mask, position_ids=position_ids)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {'do_sample': False, 'max_length': max_length}
        if eos_token_id is not None:
            generation_kwargs['eos_token_id'] = eos_token_id
            generation_kwargs['pad_token_id'] = eos_token_id # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)
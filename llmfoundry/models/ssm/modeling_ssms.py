import math
import warnings
from typing import (Any, Dict, List, Mapping, MutableMapping, Optional, Tuple,
                    Union)
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.metrics import (InContextLearningCodeEvalAccuracy,
                              InContextLearningLMAccuracy,
                              InContextLearningLMExpectedCalibrationError,
                              InContextLearningMCExpectedCalibrationError,
                              InContextLearningMultipleChoiceAccuracy,
                              InContextLearningQAAccuracy)
from composer.metrics.nlp import LanguageCrossEntropy, LanguagePerplexity
from composer.models import HuggingFaceModel
from composer.utils import dist

from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast
from llmfoundry.models.mpt.configuration_mpt import MPTConfig

from mamba_ssm.modules.mamba_simple import Block as MambaBlock
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, _init_weights
from mamba_ssm.models.config_mamba import MambaConfig
from llmfoundry.models.layers.custom_embedding import SharedEmbedding
from transformers import PretrainedConfig
import logging

log = logging.getLogger(__name__)


class MPSSMConfig(PretrainedConfig):
    def __init__(self,  d_model: int = 2560,
            n_layer: int = 64,
            vocab_size: int = 50277,
            ssm_cfg: dict ={},
            rms_norm: bool = True,
            residual_in_fp32: bool = True,
            fused_add_norm: bool = True,
            pad_vocab_size_multiple: int = 8,
            init_device: str='meta',
            **kwargs: Any):
        super().__init__(
            **kwargs,
        )
        self.d_model=d_model
        self.n_layer=n_layer
        self.vocab_size=vocab_size
        self.ssm_cfg=ssm_cfg
        self.rms_norm=rms_norm
        self.residual_in_fp32=residual_in_fp32
        self.fused_add_norm=fused_add_norm
        self.pad_vocab_size_multiple=pad_vocab_size_multiple
        self.init_device=init_device
        

class MPTPreTrainedSSMModel(PreTrainedModel):
    config_class = MPSSMConfig
    base_model_prefix = 'model'
    _no_split_modules = [MambaBlock.__name__]

class MPT_MambaLMHeadModel(MambaLMHeadModel):
    def __init__(
        self,
        config: MPSSMConfig,
        initializer_cfg: Dict=None,
        device: torch.device=None,
        dtype: torch.dtype=None,
    ) -> None:
        super().__init__(config,initializer_cfg,device,dtype)
    
    def forward(self, input_ids:Optional[torch.LongTensor], position_ids:Optional[torch.LongTensor]=None, inference_params:Dict=None, num_last_tokens:int=0):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits","hidden_states"])
        return CausalLMOutput(logits=lm_logits,hidden_states=hidden_states)
    
class MPSSMForCausalLM(MPTPreTrainedSSMModel):
    def __init__(self, config: MPSSMConfig):
        super().__init__(config)
        log.info(f'Instantiating an MPTForCausalLM model from {__file__}')

        if config.init_device == 'mixed':
            if dist.get_local_rank() == 0:
                config.init_device = 'cpu'
            else:
                config.init_device = 'meta'

        self.transformer: MPT_MambaLMHeadModel = MPT_MambaLMHeadModel(
            config=config,
            initializer_cfg=config.init_config,
            device=config.init_device,
            dtype=torch.float32
            )

        for child in self.transformer.children():
            if isinstance(child, torch.nn.ModuleList):
                continue
            if isinstance(child, torch.nn.Module):
                child._fsdp_wrap = True

        # enables scaling output logits; similar to a softmax "temperature"
        # PaLM paper uses scale 1/sqrt(config.d_model)
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(
                        f"{logit_scale=} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'."
                    )
            self.logit_scale = logit_scale
        
    def get_input_embeddings(self) -> Union[SharedEmbedding, nn.Embedding]:
        return self.transformer.backbone.embedding
    
    def get_output_embeddings(
            self) -> Union[SharedEmbedding, nn.Embedding, nn.Linear]:
        return self.transformer.get_input_embeddings()
    
    def set_input_embeddings(
            self, value: Union[SharedEmbedding, nn.Embedding]) -> None:
        self.transformer.backbone.embedding=value
        self.transformer.tie_weights()

    def set_output_embeddings(
            self, new_embeddings: Union[SharedEmbedding, nn.Embedding,
                                    nn.Linear]) -> None:
        if not isinstance(new_embeddings, (SharedEmbedding, nn.Embedding)):
                raise ValueError(
                    'new_embeddings must be an instance of SharedEmbedding ' +
                    f'or nn.Embedding, but got {type(new_embeddings)}.')
        warnings.warn(
            'Using `set_output_embeddings` to set the embedding layer of ' +
            'MPTForCausalLM with tied weights. Given weights are tied, ' +
            'using `set_input_embeddings` is recommended over using ' +
            '`set_output_embeddings`.')
        self.transformer.set_input_embeddings(new_embeddings)
    
    def tie_weights(self) -> None:
        self.transformer.tie_weights()

    def set_decoder(self, decoder: MambaLMHeadModel) -> None:
        self.transformer = decoder
    
    def get_decoder(self) -> MambaLMHeadModel:
        return self.transformer
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.FloatTensor]=None,
        num_last_tokens: Optional[int]=0, 
        inference_params: Optional[Dict]=0, 
        **kwargs: Any
    ) -> CausalLMOutputWithPast:

        outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            inference_params=inference_params, 
            num_last_tokens=num_last_tokens,
        )
        logits=outputs.logits
        
        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(
                    f'Multiplying logits by {self.logit_scale=}. This will produce uniform (uninformative) outputs.'
                )
            logits *= self.logit_scale

        loss = None
        if labels is not None:
            _labels = torch.roll(labels, shifts=-1)
            _labels[:, -1] = -100
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                _labels.to(logits.device).view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )
    
    # Param Initialization, needed for device='meta' fast initialization
    def param_init_fn(self, module: nn.Module) -> None:
        _init_weights(module=module,
                    n_layer=self.config.n_layer)
        

    # FSDP Wrap function
    def fsdp_wrap_fn(self, module: nn.Module) -> bool:
        return isinstance(module, MambaBlock)

    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        act_ckpt_list = getattr(self.config, 'activation_checkpointing_target',
                                None) or [MambaBlock.__name__]
        if isinstance(act_ckpt_list, str):
            act_ckpt_list = [act_ckpt_list]
        elif not isinstance(act_ckpt_list, list):
            raise ValueError(
                f'activation_checkpointing_target must be either a single string or a list, but got {type(act_ckpt_list)}'
            )

        if MambaBlock.__name__ in act_ckpt_list or MambaBlock.__name__.lower() in act_ckpt_list:
            if len(act_ckpt_list) > 1:
                log.info(
                    'Activation checkpointing '+MambaBlock.__name__+ ' only (ignoring other sub-block modules specified in activation_checkpointing_target).'
                )
            return isinstance(module, MambaBlock)

        mod_types = ()
        for mod_name in act_ckpt_list:
            if mod_name.lower() == MambaBlock.__name__.lower():
                mod_types += (MambaBlock,)
            else:
                msg = ', '.join( [MambaBlock.__name__])
                raise ValueError(
                    f'{mod_name} (specified in activation_checkpointing_target) is not a recognized option out of available options {msg}.'
                )
        return isinstance(module, mod_types)


class ComposerMPSSMCausalLM(HuggingFaceModel):
    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        resolved_om_model_config = om.to_container(om_model_config,
                                                   resolve=True)
        # hf_config = MPTConfig.from_dict(resolved_om_model_config)
        
        model = MPSSMForCausalLM(MPSSMConfig(**resolved_om_model_config))

        use_train_metrics = om_model_config.get('use_train_metrics', True)
        train_metrics = [LanguageCrossEntropy(),
                         LanguagePerplexity()] if use_train_metrics else []
        eval_metrics = [
            LanguageCrossEntropy(),
            LanguagePerplexity(),
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
            InContextLearningQAAccuracy(),
            InContextLearningCodeEvalAccuracy(),
            InContextLearningLMExpectedCalibrationError(),
            InContextLearningMCExpectedCalibrationError(),
        ]

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=True,
            allow_embedding_resizing=True,
        )

        self.n_active_params = sum(p.numel() for p in self.parameters())

        loss_fn_config = om_model_config.get('loss_fn', 'fused_crossentropy')
        if loss_fn_config == 'fused_crossentropy':
            try:
                # NOTE: The following is the original import statement from flash_attn library, which we have currently replaced with a copy pasted code from the same library's version 1.0.9. The reason is that using the CE loss from FA v2.3.2 results in an illegal memory access error at long sequence lengths (github issue: https://github.com/Dao-AILab/flash-attention/issues/714).
                # from flash_attn.losses.cross_entropy import \
                #     CrossEntropyLoss as FusedCrossEntropyLoss
                # TODO: Once the problem with using FA v2's CE loss at longer sequence lengths is resolved (github issue: https://github.com/Dao-AILab/flash-attention/issues/714), revert back to directly importing the CE loss from FA library.
                from llmfoundry.models.layers.cross_entropy_loss import \
                    CrossEntropyLoss as FusedCrossEntropyLoss

                self.loss_fn = FusedCrossEntropyLoss(ignore_index=-100)
            except:
                raise ValueError(
                    'Fused Cross Entropy is not installed. Either (1) have a CUDA-compatible GPU '
                    +
                    'and `pip install .[gpu]` if installing from source or `pip install xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v1.0.3#subdirectory=csrc/xentropy` '
                    +
                    'if installing from pypi, or (2) set your config model.loss_fn=torch_crossentropy.'
                )
        elif loss_fn_config == 'torch_crossentropy':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            raise ValueError(
                f'Specified loss_fn={self.loss_fn} not recognized. `loss_fn` must be one of [`fused_crossentropy`, `torch_crossentropy`].'
            )
        
    def get_targets(self, batch: Mapping) -> torch.Tensor:
        targets = torch.roll(batch['labels'], shifts=-1)
        targets[:, -1] = -100
        return targets

    def forward(self, batch: MutableMapping) -> CausalLMOutputWithPast:
        return self.model(
            input_ids=batch.get('input_ids', None),
            attention_mask=batch.get('attention_mask', None),
            prefix_mask=batch.get('bidirectional_mask', None),
            sequence_id=batch.get('sequence_id', None),
            inputs_embeds=batch.get('inputs_embeds', None),
        )

    def loss(self, outputs: CausalLMOutputWithPast,
             batch: Mapping) -> torch.Tensor:
        targets = self.get_targets(batch)
        return self.loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)),
                            targets.view(-1))

    def flops_per_batch(self, batch: Mapping) -> int:
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass

        bs, msl = batch['input_ids'].shape[0:2]
        params = self.n_active_params
        if not self.model.transformer.config.tie_word_embeddings:
            # embedding layers are lookup tables, therefore are not counted in the FLOP computation
            params -= self.model.transformer.wte.weight.numel()
        params_flops_per_token = 2 * params
        params_flops_per_seq = params_flops_per_token * msl
        attn_flops_per_seq = (self.model.config.n_layers * 2 * 2 *
                              (self.model.config.d_model * (msl**2)))

        return (params_flops_per_seq + attn_flops_per_seq) * 3 * bs    
    

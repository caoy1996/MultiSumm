# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict

from collections import OrderedDict

from fairseq import utils
from fairseq import checkpoint_utils
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture, FairseqMultiModel
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    Embedding,
    base_architecture as transformer_base_architecture,
)


@register_model("multilingual_transformer_for_summarization")
class MultilingualTransformerForSummarization(FairseqMultiModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-checkpoint",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--init-encoder -only",
            action="store_true",
        )
        parser.add_argument(
            "--init-decoder-only",
            action="store_true",
        )
        parser.add_argument('--share-encoder-embeddings', action='store_true',
                            help='share encoder embeddings across languages')
        parser.add_argument('--share-decoder-embeddings', action='store_true',
                            help='share decoder embeddings across languages')
        parser.add_argument('--share-encoders', action='store_true',
                            help='share encoders across languages')
        parser.add_argument('--share-decoders', action='store_true',
                            help='share decoders across languages')

    @classmethod
    def build_model(self, args, task, cls_dictionary=MaskedLMDictionary):
        """Build a new model instance."""
        from fairseq.tasks.multilingual_summarization import MultilingualSummarization
        assert isinstance(task, MultilingualSummarization)
        assert hasattr(args, "pretrained_checkpoint"), (
            "You must specify a path for --pretrained-checkpoint to use "
        )
        assert isinstance(task.source_dictionary, cls_dictionary) and isinstance(
            task.target_dictionary, cls_dictionary
        ), (
            "You should use a MaskedLMDictionary when using --arch. "
        )
        assert not (
            getattr(args, "init_encoder_only", False)
            and getattr(args, "init_decoder_only", False)
        ), "Only one of --init-encoder-only and --init-decoder-only can be set."

        # return super().build_model(args, task)

        # make sure all arguments are present in older models
        multilingual_base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = args.tokens_per_sample
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = args.tokens_per_sample

        # args.max_source_positions = 256
        # args.max_target_positions = 256

        src_langs = [lang_pair.split('-')[0] for lang_pair in task.model_lang_pairs]
        tgt_langs = [lang_pair.split('-')[1] for lang_pair in task.model_lang_pairs]

        if args.share_encoders:
            args.share_encoder_embeddings = True
        if args.share_decoders:
            args.share_decoder_embeddings = True

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # build shared embeddings (if applicable)
        shared_encoder_embed_tokens, shared_decoder_embed_tokens = None, None
        if args.share_all_embeddings:
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embed_dim=args.encoder_embed_dim,
                build_embedding=build_embedding,
                pretrained_embed_path=args.encoder_embed_path,
            )
            shared_decoder_embed_tokens = shared_encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            if args.share_encoder_embeddings:
                shared_encoder_embed_tokens = (
                    FairseqMultiModel.build_shared_embeddings(
                        dicts=task.dicts,
                        langs=src_langs,
                        embed_dim=args.encoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=args.encoder_embed_path,
                    )
                )
            if args.share_decoder_embeddings:
                shared_decoder_embed_tokens = (
                    FairseqMultiModel.build_shared_embeddings(
                        dicts=task.dicts,
                        langs=tgt_langs,
                        embed_dim=args.decoder_embed_dim,
                        build_embedding=build_embedding,
                        pretrained_embed_path=args.decoder_embed_path,
                    )
                )

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}

        def get_encoder(lang):
            if lang not in lang_encoders:
                if shared_encoder_embed_tokens is not None:
                    encoder_embed_tokens = shared_encoder_embed_tokens
                else:
                    encoder_embed_tokens = build_embedding(
                        task.dicts[lang], args.encoder_embed_dim, args.encoder_embed_path
                    )
                lang_encoders[lang] = TransformerEncoderFromPretrainedModel(args, task.dicts[lang], encoder_embed_tokens)
            return lang_encoders[lang]

        def get_decoder(lang):
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                else:
                    decoder_embed_tokens = build_embedding(
                        task.dicts[lang], args.decoder_embed_dim, args.decoder_embed_path
                    )
                lang_decoders[lang] = TransformerDecoderFromPretrainedModel(args, task.dicts[lang], decoder_embed_tokens)
            return lang_decoders[lang]

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if args.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src, tgt in zip(task.model_lang_pairs, src_langs, tgt_langs):
            encoders[lang_pair] = shared_encoder if shared_encoder is not None else get_encoder(src)
            decoders[lang_pair] = shared_decoder if shared_decoder is not None else get_decoder(tgt)

        return MultilingualTransformerForSummarization(encoders, decoders)

    def load_state_dict(self, state_dict, strict=True):
        state_dict_subset = state_dict.copy()
        for k, _ in state_dict.items():
            assert k.startswith('models.')
            lang_pair = k.split('.')[1]
            if lang_pair not in self.models:
                del state_dict_subset[k]
        super().load_state_dict(state_dict_subset, strict=strict)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoderFromPretrainedModel(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderFromPretrainedModel(args, tgt_dict, embed_tokens)


def upgrade_state_dict_with_pretrained_weights(
    state_dict: Dict[str, Any], pretrained_checkpoint: str
) -> Dict[str, Any]:
    """
    Load weights into a Transformer encoder or decoder model.

    Args:
        state_dict: state dict for either TransformerEncoder or
            TransformerDecoder
        pretrained_checkpoint: checkpoint to load weights from

    Raises:
        AssertionError: If architecture (num layers, attention heads, etc.)
            does not match between the current Transformer encoder or
            decoder and the pretrained_checkpoint
    """
    if not os.path.exists(pretrained_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_checkpoint))

    state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_checkpoint)
    model_state_dict = state["model"]
    print(model_state_dict.keys())
    for model_key in model_state_dict.keys():

        # for search_key in ["embed_tokens", "embed_positions", "layers"]:
        for search_key in ["embed_tokens", "layers"]:
            if search_key in model_key:
                subkey = model_key[model_key.find(search_key):]
                assert subkey in state_dict, (
                    "{} Transformer encoder / decoder "
                    "state_dict does not contain {}. Cannot "
                    "load {} from pretrained checkpoint "
                    "{} into Transformer.".format(
                        str(state_dict.keys()),
                        subkey, model_key, pretrained_checkpoint)
                    )

                state_dict[subkey] = model_state_dict[model_key]
    return state_dict


class TransformerEncoderFromPretrainedModel(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, 'init_decoder_only', False):
            # Don't load weights for encoder if --init-decoder-only
            return

        assert hasattr(args, "pretrained_checkpoint"), (
            "--pretrained-checkpoint must be specified to load Transformer "
            "encoder from pretrained model"
        )
        model_loaded_state_dict = upgrade_state_dict_with_pretrained_weights(
            state_dict=self.state_dict(),
            pretrained_checkpoint=args.pretrained_checkpoint,
        )
        self.load_state_dict(model_loaded_state_dict, strict=False)


class TransformerDecoderFromPretrainedModel(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if getattr(args, 'init_encoder_only', False):
            # Don't load weights for decoder if --init-encoder-only
            return
        assert hasattr(args, "pretrained_checkpoint"), (
            "--pretrained-checkpoint must be specified to load Transformer "
            "decoder from pretrained model"
        )

        model_loaded_state_dict = upgrade_state_dict_with_pretrained_weights(
            state_dict=self.state_dict(),
            pretrained_checkpoint=args.pretrained_checkpoint,
        )
        self.load_state_dict(model_loaded_state_dict, strict=False)


@register_model_architecture(
    "multilingual_transformer_for_summarization", "multilingual_transformer_for_summarization"
)
def multilingual_base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.share_encoder_input_output_embed = getattr(
        args, 'share_encoder_input_output_embed', True)
    args.no_token_positional_embeddings = getattr(
        args, 'no_token_positional_embeddings', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.num_segment = getattr(args, 'num_segment', 1)

    args.encoder_layers = getattr(args, 'encoder_layers', 6)

    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.bias_kv = getattr(args, 'bias_kv', False)
    args.zero_attn = getattr(args, 'zero_attn', False)

    args.sent_loss = getattr(args, 'sent_loss', False)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.apply_bert_init = getattr(args, 'apply_bert_init', True)
    transformer_base_architecture(args)
    args.share_encoder_embeddings = getattr(args, 'share_encoder_embeddings', True)
    args.share_decoder_embeddings = getattr(args, 'share_decoder_embeddings', True)
    args.share_encoders = getattr(args, 'share_encoders', True)
    args.share_decoders = getattr(args, 'share_decoders', True)

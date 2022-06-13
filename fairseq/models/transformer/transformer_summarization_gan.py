# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.dataclass.utils import gen_parser_from_dataclass

from fairseq.models.transformer.transformer_base import (
    TransformerModelBase,
)

from fairseq.models.transformer import (
    base_architecture,
    TransformerConfig,
)

from distutils.util import strtobool
from decoding_algorithms.ctc_scope_search_length_control import CTCScopeSearchLengthControlDecoder
from decoding_algorithms.at_beam_search_char_length_control import ATScopeSearchCharLengthControlDecoder


@register_model("transformer_summarization_gan")
class TransformerModelSummarizationGan(TransformerModelBase):
    """
    This is the implementation of the transformer model for summarization
    """

    def __init__(self, cfg, encoder_rec, decoder_rec, encoder_hc, decoder_hc):
        super().__init__(cfg, encoder_rec, decoder_hc)
        self.bottleneck_type = cfg.bottleneck_type
        self.bottleneck = None
        self.build_bottleneck()
        self.encoder_rec = encoder_rec
        self.decoder_rec = decoder_rec
        self.encoder_hc = encoder_hc
        self.decoder_hc = decoder_hc
        # TODO: Change the forward function and add some information bottlenecks.

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )
        parser.add_argument(
            '--bottleneck_type',
            default="reduce_hidden_state_num",
            type=str,
            choices=["reduce_hidden_state_num"],
            help="Options to control the information bottleneck",
        )

    def build_bottleneck(self):
        if self.bottleneck_type == "bottleneck_type":
            decoder_input_dim = self.cfg.decoder.input_dim
            pass
            self.bottleneck = None

    @classmethod
    def build_model(cls, args, task):
        """
        Build two Transformer models.
        One model is going to be trained by reconstruction loss while the other model will be trained by HC summary
        """
        summarization_architecture(args)
        cfg = TransformerConfig.from_namespace(args)

        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        # TODO: design some parameters to control embedding sharing, we are currently forcing sharing all parameters
        if src_dict != tgt_dict:
            raise ValueError("--share-all-embeddings requires a joined dictionary")
        if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
            raise ValueError("--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim")
        if cfg.decoder.embed_path and (cfg.decoder.embed_path != cfg.encoder.embed_path):
            raise ValueError("--share-all-embeddings not compatible with --decoder-embed-path")
        encoder_rec_embed_tokens = cls.build_embedding(cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path)
        decoder_rec_embed_tokens = encoder_rec_embed_tokens
        encoder_hc_embed_tokens = encoder_rec_embed_tokens
        decoder_hc_embed_tokens = encoder_hc_embed_tokens
        cfg.share_decoder_input_output_embed = True

        encoder_rec = cls.build_encoder(cfg, src_dict, encoder_rec_embed_tokens)
        decoder_rec = cls.build_decoder(cfg, tgt_dict, decoder_rec_embed_tokens)
        encoder_hc = cls.build_encoder(cfg, src_dict, encoder_hc_embed_tokens)
        decoder_hc = cls.build_decoder(cfg, tgt_dict, decoder_hc_embed_tokens)

        return cls(cfg, encoder_rec, decoder_rec, encoder_hc, decoder_hc)


@register_model_architecture("transformer_summarization", "transformer_summarization")
def summarization_architecture(args):
    base_architecture(args)



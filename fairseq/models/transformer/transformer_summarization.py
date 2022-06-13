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


@register_model("transformer_summarization")
class TransformerModelSummarization(TransformerModelBase):
    """
    This is the implementation of the transformer model for summarization
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        self.ctc_decoder = self.create_ctc_decoder(cfg)

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )
        # Add some summarization related stuff.
        parser.add_argument(
            '--decoding_algorithm',
            default="beam_search",
            type=str,
            choices=["beam_search", "char_beam_search", "ctc_beam_search_length_control",
                     "ctc_beam_search_char_length_control"],

            help="Options to control the the decoding method",
        )
        parser.add_argument(
            '--truncate_summary',
            default=False,
            type=strtobool,
            help="Whether to truncate the generated summaries. Notice this is only valid for ctc_greedy decoding "
                 "and ctc_beam_search",
        )
        parser.add_argument(
            '--force_length',
            default=False,
            type=strtobool
        )
        parser.add_argument(
            '--desired_length',
            default=10,
            type=int
        )
        parser.add_argument(
            '--use_length_ratio',
            default=False,
            type=strtobool
        )
        parser.add_argument(
            '--k',
            default=10,
            type=int
        )
        parser.add_argument(
            '--beam_size',
            default=6,
            type=int
        )
        parser.add_argument(
            '--marg_criteria',
            default="max",
            type=str
        )
        parser.add_argument(
            '--scope',
            default=3,
            type=int
        )
        parser.add_argument(
            '--scaling_factor',
            default=4,
            type=int
        )

    def create_ctc_decoder(self, args):
        """
        Create a CTC decoder to map logits to words, based on the user-specified decoding choice
        """
        ctc_decoding_algorithm = getattr(args, 'decoding_algorithm')

        if ctc_decoding_algorithm == "ctc_beam_search_length_control":
            assert getattr(args, 'truncate_summary') == False, "Cannot truncate summary with length control decoding"
            decoding_params = {
                'force_length': getattr(args, 'force_length'),
                "desired_length": getattr(args, 'desired_length'),
                "use_length_ratio": getattr(args, 'use_length_ratio'),
                "k": getattr(args, 'beam_size'),
                "beam_size": getattr(args, 'beam_size'),
                "scope": getattr(args, 'scope'),
                "marg_criteria": getattr(args, 'marg_criteria', 'max'),
                # truncate_summary is a dummy variable since length control does not need it
                "truncate_summary": getattr(args, 'truncate_summary')
            }
            decoder = CTCScopeSearchLengthControlDecoder(self.encoder.dictionary, decoding_params)

        elif ctc_decoding_algorithm == "ctc_beam_search_char_length_control":
            assert getattr(args, 'truncate_summary') == False, "Cannot truncate summary with length control decoding"
            decoding_params = {
                'force_length': getattr(args, 'force_length'),
                "desired_length": getattr(args, 'desired_length'),
                "use_length_ratio": getattr(args, 'use_length_ratio'),
                "k": getattr(args, 'k'),
                "beam_size": getattr(args, 'beam_size'),
                "scope": getattr(args, 'scope'),
                "marg_criteria": getattr(args, 'marg_criteria', 'max'),
                # truncate_summary is a dummy variable since length control does not need it
                "truncate_summary": getattr(args, 'truncate_summary'),
                "scaling_factor": getattr(args, 'scaling_factor'),
            }
            decoder = ATScopeSearchCharLengthControlDecoder(self.encoder.dictionary, decoding_params)
        else:
            decoder = None
        return decoder

    @classmethod
    def build_model(cls, args, task):
        summarization_architecture(args)
        cfg = TransformerConfig.from_namespace(args)
        return super().build_model(cfg, task)


@register_model_architecture("transformer_summarization", "transformer_summarization")
def summarization_architecture(args):
    base_architecture(args)

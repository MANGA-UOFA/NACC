# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import utils
from distutils.util import strtobool

from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat.nonautoregressive_transformer import NATransformerDecoder
from fairseq.models.nat import FairseqNATModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import logging
import torch

try:
    # Decoder for naive ctc beam search, not runnable on Compute Canada
    from decoding_algorithms.ctc_beam_search import CTCBeamSearchDecoder
except:
    CTCBeamSearchDecoder = None
    print("Failed to load CTCBeamSearchDecoder, ignore this message if you are using Compute Canada. \n")
# Decoders for word-level length control
from decoding_algorithms.ctc_greedy_decoding import CTCGreedyDecoder
from decoding_algorithms.ctc_beam_search_length_control import CTCBeamSearchLengthControlDecoder
from decoding_algorithms.ctc_scope_search_length_control import CTCScopeSearchLengthControlDecoder
# Decoders for char length control
from decoding_algorithms.ctc_char_greedy_decoding import CTCCharGreedyDecoder
from decoding_algorithms.ctc_beam_search_char_length_control import CTCBeamSearchCharLengthControlDecoder
from decoding_algorithms.ctc_scope_search_char_length_control import CTCScopeSearchCharLengthControlDecoder
from decoding_algorithms.ctc_test_decoding import CTCTestDecoder


logger = logging.getLogger(__name__)


@register_model("nat_encoder_decoder_ctc")
class NATransformerEncoderOnlyModel(FairseqNATModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.ctc_decoder = self.create_ctc_decoder(args)  # This is not Transformer decoder!
        self.cfg = args

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser, **kwargs):
        FairseqNATModel.add_args(parser)
        parser.add_argument(
            '--decoding_algorithm',
            default="ctc_greedy_decoding",
            type=str,
            choices=["ctc_greedy_decoding", "ctc_beam_search", "ctc_beam_search_length_control",
                     "ctc_beam_search_char_length_control", "ctc_char_greedy_decoding",
                     "ctc_scope_search_length_control", "ctc_scope_search_char_length_control",
                     "ctc_test_length_control"],

            help="Options to control the the CTC decoding method",
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
        parser.add_argument(
            '--encoder_layers',
            default=6,
            type=int
        )

    def decode_lprob_to_token_index(self, lprobs, sample=None):
        if sample is None:
            source_length = None
        else:
            source_length = sample["net_input"]["src_lengths"]
        return self.ctc_decoder.decode(lprobs, source_length=source_length)

    def create_ctc_decoder(self, args):
        """
        Create a CTC decoder to map logits to words, based on the user-specified decoding choice
        """
        decoding_algorithm = getattr(args, 'decoding_algorithm')
        if decoding_algorithm == "ctc_greedy_decoding":
            assert getattr(args, 'force_length') == False, "Cannot force length for greedy decoding"
            decoding_params = {"truncate_summary": getattr(args, 'truncate_summary'),
                               "desired_length": getattr(args, 'desired_length'), }
            decoder = CTCGreedyDecoder(self.encoder.dictionary, decoding_params)
        elif decoding_algorithm == "ctc_beam_search":
            assert getattr(args, 'force_length') == False, "Cannot force length for ctc naive beam search"
            decoding_params = {
                "truncate_summary": getattr(args, 'truncate_summary'),
                "desired_length": getattr(args, 'desired_length'),
                "model_path": None,
                "alpha": 0,
                "beta": 0,
                "cutoff_top_n": getattr(args, 'k'),
                "cutoff_prob": 1.0,
                "beam_width": getattr(args, 'beam_size'),
                "num_processes": 4,
                "log_probs_input": True}
            decoder = CTCBeamSearchDecoder(self.encoder.dictionary, decoding_params)
            print("Successfully created the CTC Beam search decoder")
        elif decoding_algorithm == "ctc_beam_search_length_control":
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
            # decoding_params = {
            #     'force_length': getattr(args, 'force_length'),
            #     "desired_length": getattr(args, 'desired_length'),
            #     "use_length_ratio": getattr(args, 'use_length_ratio'),
            #     "k": getattr(args, 'k'),
            #     "beam_size": getattr(args, 'beam_size'),
            #     # truncate_summary is a dummy variable since length control does not need it
            #     "truncate_summary": getattr(args, 'truncate_summary')
            # }
            # decoder = CTCBeamSearchLengthControlDecoder(self.encoder.dictionary, decoding_params)
        elif decoding_algorithm == "ctc_scope_search_length_control":
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
        elif decoding_algorithm == "ctc_char_greedy_decoding":
            assert getattr(args, 'force_length') == False, "Cannot force length for greedy decoding"
            decoding_params = {"truncate_summary": getattr(args, 'truncate_summary'),
                               "desired_length": getattr(args, 'desired_length'), }
            decoder = CTCCharGreedyDecoder(self.encoder.dictionary, decoding_params)
        elif decoding_algorithm == "ctc_beam_search_char_length_control":
            assert getattr(args, 'truncate_summary') == False, "Cannot truncate summary with length control decoding"
            # decoding_params = {
            #     'force_length': getattr(args, 'force_length'),
            #     "desired_length": getattr(args, 'desired_length'),
            #     "use_length_ratio": getattr(args, 'use_length_ratio'),
            #     "k": getattr(args, 'k'),
            #     "beam_size": getattr(args, 'beam_size'),
            #     # truncate_summary is a dummy variable since length control does not need it
            #     "truncate_summary": getattr(args, 'truncate_summary')
            # }
            # decoder = CTCBeamSearchCharLengthControlDecoder(self.encoder.dictionary, decoding_params)
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
            decoder = CTCScopeSearchCharLengthControlDecoder(self.encoder.dictionary, decoding_params)
        elif decoding_algorithm == "ctc_scope_search_char_length_control":
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
            decoder = CTCScopeSearchCharLengthControlDecoder(self.encoder.dictionary, decoding_params)
        elif decoding_algorithm == "ctc_test_length_control":
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
            decoder = CTCTestDecoder(self.encoder.dictionary, decoding_params)
        else:
            raise NotImplementedError("%s is not supported" % decoding_algorithm)
        return decoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(self, sample, something_else=None):
        """
        Forward function of the model
        """
        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        src_lengths = net_input["src_lengths"]
        prev_output_tokens = torch.zeros(src_tokens.shape, dtype=torch.long, device=src_tokens.device)
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        word_ins_out = self.decoder(normalize=False, prev_output_tokens=prev_output_tokens, encoder_out=encoder_out)

        return word_ins_out

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output
        if log_probs:
            return logits.log_softmax(dim=-1)
        else:
            return logits.softmax(dim=-1)


@register_model_architecture(
    "nat_encoder_decoder_ctc", "nat_encoder_decoder_ctc"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)  # NOTE

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)

    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

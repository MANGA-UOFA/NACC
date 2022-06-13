# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from distutils.util import strtobool
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATModel
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from typing import Union
import logging
import random, math
from .nat_sd_shared import NATransformerDecoder

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


@register_model("nat_encoder_decoder_ctc_sd")
class NATransformerEncoderDecoderSDModel(FairseqNATModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.inference_decoder_layer = getattr(args, 'inference_decoder_layer', -1)
        self.ctc_decoder = self.create_ctc_decoder(args)  # This is not Transformer decoder!

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        def add_chenyang_param():
            parser.add_argument(
                "--src-embedding-copy",
                action="store_true",
                help="copy encoder word embeddings as the initial input of the decoder",
            )
            parser.add_argument(
                "--pred-length-offset",
                action="store_true",
                help="predicting the length difference between the target and source sentences",
            )
            parser.add_argument(
                "--sg-length-pred",
                action="store_true",
                help="stop the gradients back-propagated from the length predictor",
            )
            parser.add_argument(
                "--length-loss-factor",
                type=float,
                help="weights on the length prediction loss",
            )
            parser.add_argument(
                "--src-upsample-scale",
                type=int,
                default=1
            )
            parser.add_argument(
                '--use-ctc-decoder',
                action='store_true',
                default=False
            )
            parser.add_argument(
                '--ctc-beam-size',
                default=1,
                type=int
            )
            parser.add_argument(
                '--ctc-beam-size-train',
                default=1,
                type=int
            )
            # parser.add_argument(
            #     '--hard-argmax',
            #     action='store_true',
            #     default=False
            # )
            # parser.add_argument(
            #     '--yhat-temp',
            #     type=float,
            #     default=0.1
            # )
            parser.add_argument(
                '--inference-decoder-layer',
                type=int,
                default=-1
            )
            parser.add_argument(
                '--share-ffn',
                action='store_true',
                default=False
            )
            parser.add_argument(
                '--share-attn',
                action='store_true',
                default=False
            )
            parser.add_argument(
                '--sample-option',
                type=str,
                default='hard'
            )
            parser.add_argument(
                '--softmax-temp',
                type=float,
                default=1
            )
            parser.add_argument(
                '--temp-anneal',
                action='store_true',
                default=False
            )
            parser.add_argument(
                '--num-topk',
                default=1,
                type=int
            )
            parser.add_argument(
                '--copy-src-token',
                action='store_true',
                default=False
            )
            parser.add_argument(
                '--force-detach',
                action='store_true',
                default=False
            )
            parser.add_argument(
                '--softcopy',
                action='store_true',
                default=False
            )
            parser.add_argument(
                '--softcopy-temp',
                default=5,
                type=float
            )
            parser.add_argument(
                '--concat-yhat',
                action='store_true',
                default=False
            )
            parser.add_argument(
                '--concat-dropout',
                type=float,
                default=0
            )
            parser.add_argument(
                '--layer-drop-ratio',
                type=float,
                default=0.0
            )
            parser.add_argument(
                '--all-layer-drop',
                action='store_true',
                default=False
            )
            parser.add_argument(
                '--yhat-posemb',
                action='store_true',
                default=False
            )
            parser.add_argument(
                '--dropout-anneal',
                action='store_true',
                default=False
            )
            parser.add_argument(
                '--dropout-anneal-end-ratio',
                type=float,
                default=0
            )
            parser.add_argument(
                '--force-ls',
                action='store_true',
                default=False
            )
            parser.add_argument(
                '--repeat-layer',
                type=int,
                default=0
            )

        def add_length_control_param():
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

        add_chenyang_param()
        add_length_control_param()

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        decoder.repeat_layer = getattr(args, 'repeat_layer', 0)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

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
            raise (NotImplementedError, "%s is not supported" % decoding_algorithm)
        return decoder

    def forward(self, sample, something_else=None, train_ratio=None):

        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        src_lengths = net_input["src_lengths"]
        prev_output_tokens = self.initialize_output_tokens_by_src_tokens(src_tokens)
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)

        # decoding
        output_logits_list = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            train_ratio=train_ratio
        )

        # return output_logits_list
        if self.training:
            # Return the prediction of each layer during training to calculate the process
            return output_logits_list
        else:
            return output_logits_list[self.inference_decoder_layer]

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output
        if log_probs:
            return logits.log_softmax(dim=-1)
        else:
            return logits.softmax(dim=-1)

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        # set CTC decoder beam size
        self.ctc_decoder._beam_width = self.args.ctc_beam_size
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        history = decoder_out.history

        # execute the decoder
        output_logits_list = self.decoder(
            normalize=False,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        )

        inference_decoder_layer = self.inference_decoder_layer
        output_logits = output_logits_list[inference_decoder_layer]

        if self.plain_ctc:
            output_scores = decoder_out.output_scores
            _scores, _tokens = output_logits.max(-1)
            output_masks = output_tokens.ne(self.pad)
            output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
            output_scores.masked_scatter_(output_masks, _scores[output_masks])
            if history is not None:
                history.append(output_tokens.clone())

            return decoder_out._replace(
                output_tokens=output_tokens,
                output_scores=output_scores,
                attn=None,
                history=history,
            )
        else:
            # _scores, _tokens = F.log_softmax(output_logits, -1).max(-1)
            # _scores == beam_results[:,0,:]
            output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1)
            beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(F.softmax(output_logits, -1),
                                                                                     output_length)
            top_beam_tokens = beam_results[:, 0, :]
            top_beam_len = out_lens[:, 0]
            mask = torch.arange(0, top_beam_tokens.size(1)).type_as(top_beam_len). \
                repeat(top_beam_len.size(0), 1).lt(top_beam_len.unsqueeze(1))
            top_beam_tokens[~mask] = self.decoder.dictionary.pad()
            # output_scores.masked_scatter_(output_masks, _scores[output_masks])
            if history is not None:
                history.append(output_tokens.clone())

            return decoder_out._replace(
                output_tokens=top_beam_tokens.to(output_logits.device),
                output_scores=torch.full(top_beam_tokens.size(), 1.0),
                attn=None,
                history=history,
            )

    def initialize_output_tokens_by_src_tokens(self, src_tokens):
        if not self.args.copy_src_token:
            length_tgt = torch.sum(src_tokens.ne(self.tgt_dict.pad_index), -1)
            if self.args.src_upsample_scale > 2:
                length_tgt = length_tgt * self.args.src_upsample_scale
            else:
                length_tgt = length_tgt * self.args.src_upsample_scale  # + 10
            max_length = length_tgt.clamp_(min=2).max()
            idx_length = utils.new_arange(src_tokens, max_length)

            initial_output_tokens = src_tokens.new_zeros(
                src_tokens.size(0), max_length
            ).fill_(self.pad)
            initial_output_tokens.masked_fill_(
                idx_length[None, :] < length_tgt[:, None], self.unk
            )
            initial_output_tokens[:, 0] = self.bos
            initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
            return initial_output_tokens
        else:
            if self.args.src_upsample_scale <= 1:
                return src_tokens

            def _us(x, s):
                B = x.size(0)
                _x = x.unsqueeze(-1).expand(B, -1, s).reshape(B, -1)
                return _x

            return _us(src_tokens, self.args.src_upsample_scale)

    def initialize_output_tokens(self, encoder_out, src_tokens):
        # length prediction
        initial_output_tokens = self.initialize_output_tokens_by_src_tokens(src_tokens)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

@register_model_architecture(
    "nat_encoder_decoder_ctc_sd", "nat_encoder_decoder_ctc_sd"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)  # NOTE
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)  # NOTE
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)

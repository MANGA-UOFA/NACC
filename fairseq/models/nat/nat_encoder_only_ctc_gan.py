# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from distutils.util import strtobool

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer.transformer_base import Embedding
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.transformer.transformer_config import TransformerConfig
from fairseq.models.fairseq_model import BaseFairseqModel
from fairseq.models.nat.fairseq_nat_model import FairseqNATEncoder
from fairseq.models.transformer import TransformerDecoder, TransformerEncoder


try:
    # Decoder for naive ctc beam search, not runnable on Compute Canada
    from decoding_algorithms.ctc_beam_search import CTCBeamSearchDecoder
except:
    CTCBeamSearchDecoder = None
    print("Failed to load CTCBeamSearchDecoder, ignore this message if you are using Compute Canada. \n")
# Decoders for word-level length control
from decoding_algorithms.ctc_greedy_decoding import CTCGreedyDecoder
from decoding_algorithms.ctc_scope_search_length_control import CTCScopeSearchLengthControlDecoder
# Decoders for char length control
from decoding_algorithms.ctc_char_greedy_decoding import CTCCharGreedyDecoder
from decoding_algorithms.ctc_scope_search_char_length_control import CTCScopeSearchCharLengthControlDecoder
from decoding_algorithms.ctc_test_decoding import CTCTestDecoder

logger = logging.getLogger(__name__)


@register_model("nat_encoder_only_ctc_gan")
class NATransformerEncoderOnlyGANModel(BaseFairseqModel):
    """
        This is model trains the encoder output to match
    """

    def __init__(self, args, encoder_ctc, encoder_rec, decoder_rec, classifier):
        super().__init__()
        self.encoder_rec = encoder_rec  # Reconstruction encoder
        self.decoder_rec = decoder_rec  # Reconstruction decoder
        self.encoder_ctc = encoder_ctc
        self.classifier = classifier
        self.ctc_decoder = self.create_ctc_decoder(args)  # This is not Transformer decoder!
        self.cfg = args

        def enc_output_projection_ctc(encoder_ctc_out):
            latent_space = encoder_ctc_out["encoder_out"][0].transpose(0, 1)
            return torch.matmul(latent_space, self.encoder_ctc.embed_tokens.weight.transpose(0, 1))

        def enc_output_projection_rec(encoder_rec_out):
            latent_space = encoder_rec_out["encoder_out"][0].transpose(0, 1)
            return torch.matmul(latent_space, self.encoder_rec.embed_tokens.weight.transpose(0, 1))

        self.enc_output_projection_ctc = enc_output_projection_ctc
        self.enc_output_projection_rec = enc_output_projection_rec

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser, **kwargs):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=True, with_prefix=""
        )
        parser.add_argument(
            "--apply-bert-init",
            action="store_true",
            help="use custom param initialization for BERT",
        )
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
            decoder = CTCGreedyDecoder(self.encoder_ctc.dictionary, decoding_params)
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
            decoder = CTCBeamSearchDecoder(self.encoder_ctc.dictionary, decoding_params)
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
            decoder = CTCScopeSearchLengthControlDecoder(self.encoder_ctc.dictionary, decoding_params)
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
            decoder = CTCScopeSearchLengthControlDecoder(self.encoder_ctc.dictionary, decoding_params)
        elif decoding_algorithm == "ctc_char_greedy_decoding":
            assert getattr(args, 'force_length') == False, "Cannot force length for greedy decoding"
            decoding_params = {"truncate_summary": getattr(args, 'truncate_summary'),
                               "desired_length": getattr(args, 'desired_length'), }
            decoder = CTCCharGreedyDecoder(self.encoder_ctc.dictionary, decoding_params)
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
            decoder = CTCScopeSearchCharLengthControlDecoder(self.encoder_ctc.dictionary, decoding_params)
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
            decoder = CTCScopeSearchCharLengthControlDecoder(self.encoder_ctc.dictionary, decoding_params)
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
            decoder = CTCTestDecoder(self.encoder_ctc.dictionary, decoding_params)
        else:
            raise (NotImplementedError, "%s is not supported" % decoding_algorithm)
        return decoder

    def forward(self, sample, something_else=None):
        """
        Forward function of the model
        """
        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        src_lengths = net_input["src_lengths"]
        encoder_rec_out = self.encoder_rec(src_tokens, src_lengths=src_lengths)
        projected_encoder_rec_out = self.enc_output_projection_rec(encoder_rec_out)
        return projected_encoder_rec_out

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output
        if log_probs:
            return logits.log_softmax(dim=-1)
        else:
            return logits.softmax(dim=-1)

    @classmethod
    def build_embedding(cls, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder_ctc(cls, args, src_dict, embed_tokens):
        encoder_ctc = FairseqNATEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder_ctc.apply(init_bert_params)
        return encoder_ctc

    @classmethod
    def build_encoder_rec(cls, args, src_dict, embed_tokens):
        encoder_rec = TransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder_rec.apply(init_bert_params)
        return encoder_rec

    @classmethod
    def build_decoder_rec(cls, args, tgt_dict, embed_tokens):
        decoder_rec = TransformerDecoder(args, tgt_dict, embed_tokens)
        return decoder_rec

    @classmethod
    def build_classifier(cls, args):
        class EncoderOutputClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(args.encoder_embed_dim, 200)
                self.fc2 = nn.Linear(200, 200)
                self.fc3 = nn.Linear(200, 20)
                self.fc4 = nn.Linear(20, 2)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = torch.softmax(self.fc4(x), dim=-1)
                return x

        return EncoderOutputClassifier()

    @classmethod
    def build_model(cls, args, task):
        """
        Instead of directly instancing the class through initialization, fairseq needs an extra
        class function build_model to build an instance of the given class.
        """
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        encoder_ctc_embed_tokens = cls.build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
        # encoder_rec_embed_tokens = cls.build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
        else:
            raise ValueError("--share-all-embeddings must be True for this encoder-only model")
        # if cfg.offload_activations:
        #     cfg.checkpoint_activations = True  # offloading implies checkpointing
        # TODO: We are currently share all embeddings, may need a parameter to control the embedding share
        encoder_ctc = cls.build_encoder_ctc(args, src_dict, encoder_ctc_embed_tokens)
        encoder_rec = cls.build_encoder_rec(args, src_dict, encoder_ctc_embed_tokens)
        decoder_rec = cls.build_decoder_rec(args, src_dict, encoder_ctc_embed_tokens)
        classifier = cls.build_classifier(args)
        return cls(args, encoder_ctc, encoder_rec, decoder_rec, classifier)

    @property
    def has_encoder(self):
        """
        Property function which tells whether the model has an encoder
        """
        return True

    @property
    def has_decoder(self):
        """
        Property function which tells whether the model has a decoder
        """
        return True

    @staticmethod
    def normalize_and_standardize(hidden_state):
        """
        Normalize and then standardize the hidden state
        """
        # Normalize
        hidden_state = hidden_state/hidden_state.max(-1)
        # Standardize
        means = hidden_state.mean(dim=-1, keepdim=True)
        stds = hidden_state.std(dim=-1, keepdim=True)
        normalized_hidden_state = (hidden_state - means) / stds
        return normalized_hidden_state


@register_model_architecture(
    "nat_encoder_only_ctc_gan", "nat_encoder_only_ctc_gan"
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

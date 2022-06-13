# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import utils
from distutils.util import strtobool

from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat.fairseq_nat_encoder_only_model import FairseqNATEncoderOnly
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
                (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t


def get_binary_target(source_batch, target_batch, source_length_batch):
    """
    Find the binary representation of the target. i.e., if a word in the source is kept in the target,
    we set the binary target to 1, otherwise 0.
    Note the tokens in the target are extracted from the source with order preserved.
    The input source_batch is a Tensor, the target batch is also a Tensor.
    This model was designed to demonstrate the NAUS is better than a simple binary classifier.
    """
    sample_num = len(source_batch)
    binary_target = torch.zeros(source_batch.shape, dtype=torch.long, device=source_batch.device) - 1
    for i in range(0, sample_num):
        # for each sample
        current_source = source_batch[i]
        current_target = target_batch[i]
        # We have to do something to remove the padding source token.
        # e.g., leave them as -1 and ignore those -1 tokens during the cross-entropy loss calculation.
        current_source_length = source_length_batch[i]
        current_target_length = len(current_target)
        target_pointer = 0
        for j in range(0, current_source_length):
            if target_pointer == current_target_length:
                # If we have visited the last element in the target, remaining element in source musn't be selected.
                binary_target[i, j] = 0
            elif current_source[j] == current_target[target_pointer]:  # If the current word is extracted to the summary
                binary_target[i, j] = 1
                target_pointer += 1  # Move the pointer to the next word in target
            else:
                binary_target[i, j] = 0

    return binary_target


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, -1)
        return x


class BinaryClassificationDecoder:
    """
    Decode the binary probabilities into tokens
    """
    
    def __init__(self, args):
        self.truncate_summary = args.truncate_summary
        self.desired_length = args.desired_length

    def decode(self, lprob, src_tokens):
        extractive_mask = lprob.argmax(-1)
        generated_summary_list = []
        batch_size = lprob.shape[0]
        for i in range(0, batch_size):
            current_src_tokens = src_tokens[i]
            current_extractive_mask = extractive_mask[i]
            current_generated_summary = current_src_tokens[current_extractive_mask.bool()]
            current_generated_summary = current_generated_summary.tolist()
            if self.truncate_summary:
                current_generated_summary = current_generated_summary[:self.desired_length]

            generated_summary_list.append(current_generated_summary)
        return generated_summary_list


@register_model("nat_encoder_only_binary_classifier")
class NATransformerEncoderOnlyModel(FairseqNATEncoderOnly):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)
        self.binary_classifier = BinaryClassifier()
        self.binary_classifier_decoder = BinaryClassificationDecoder(args)

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser, **kwargs):
        FairseqNATEncoderOnly.add_args(parser)
        parser.add_argument(
            '--truncate_summary',
            default=False,
            type=strtobool,
            help="Whether to truncate the generated summaries. Notice this is only valid for ctc_greedy decoding "
                 "and ctc_beam_search",
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

    def decode_lprob_to_token_index(self, lprobs, sample):
        """
        Use the binary classification log-prob to determine token selection.
        """
        return self.binary_classifier_decoder.decode(lprobs, sample["net_input"]["src_tokens"])

    def forward(self, sample, something_else=None):
        """
        Forward function of the model
        """
        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        src_lengths = net_input["src_lengths"]
        tgt_tokens = sample["target"]
        encoder_output = super().forward_encoder(src_tokens, src_lengths)
        classified_result = self.binary_classifier(encoder_output["encoder_out"][0].transpose(0, 1))
        return classified_result

    def get_targets(self, sample, net_output):
        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        src_lengths = net_input["src_lengths"]
        tgt_tokens = sample["target"]
        binary_target = get_binary_target(src_tokens, tgt_tokens, src_lengths)
        return binary_target

    def initialize_output_tokens_by_src_tokens(self, src_tokens):
        if not self.copy_src_token:
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


@register_model_architecture(
    "nat_encoder_only_binary_classifier", "nat_encoder_only_binary_classifier"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)  # NOTE
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

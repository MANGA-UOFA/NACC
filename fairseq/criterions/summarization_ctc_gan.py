# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round
from fairseq.models.fairseq_model import BaseFairseqModel
from customized_ctc import ctc_loss as custom_ctc_loss


@dataclass
class SummarizationCtcGanCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="wordpiece",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    label_smoothing: float = field(
        default=0,
        metadata={
            "help": "label smoothing during the CTC loss calculation"
        }
    )
    use_customized_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to use customized ctc loss for CTC-training"
        }
    )


@register_criterion("summarization_ctc_gan", dataclass=SummarizationCtcGanCriterionConfig)
class SummarizationCtcCriterion(FairseqCriterion):
    def __init__(self, cfg: SummarizationCtcGanCriterionConfig, task: FairseqTask) -> None:
        super().__init__(task)
        self.blank_idx = task.target_dictionary.blank()
        self.bos_idx = task.target_dictionary.bos()
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg
        self.label_smoothing = cfg.label_smoothing
        self.use_customized_loss = cfg.use_customized_loss
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, model: BaseFairseqModel, sample: Dict, reduce=True):
        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        src_lengths = net_input["src_lengths"]
        # Inference of the ctc encoder
        encoder_ctc_out = model.encoder_ctc(src_tokens, src_lengths)
        encoder_ctc_projected_out = model.enc_output_projection_ctc(encoder_ctc_out)
        encoder_ctc_lprobs = model.get_normalized_probs(encoder_ctc_projected_out, log_probs=True).contiguous()

        # Inference of the reconstruction encoder and decoder
        encoder_rec_out = model.encoder_rec(src_tokens, src_lengths)
        encoder_rec_projected_out = model.enc_output_projection_rec(encoder_rec_out)
        decoder_rec_out = model.decoder_rec(sample["net_input"]["src_tokens"][:, 0:-1],
                                            encoder_out=encoder_rec_out,
                                            features_only=False,
                                            alignment_layer=None,
                                            alignment_heads=None,
                                            src_lengths=sample["net_input"]["src_lengths"],
                                            return_all_hiddens=True)

        # Calculate ctc loss
        ctc_loss = self.get_ctc_loss(encoder_ctc_lprobs, sample)

        # Calculate reconstruction loss
        vocab_size = decoder_rec_out[0].shape[-1]
        flattened_decoder_rec_out = decoder_rec_out[0].view(-1, vocab_size)
        flattened_src_token = sample["net_input"]["src_tokens"][:, 1:].flatten()
        reconstruction_loss = self.cross_entropy_loss(flattened_decoder_rec_out, flattened_src_token)

        # Calculate GAN loss
        hidden_state_size = encoder_ctc_out["encoder_out"][0].shape[-1]
        flatten_encoder_ctc_out = encoder_ctc_out["encoder_out"][0].view(-1, hidden_state_size)
        flatten_encoder_rec_out = encoder_rec_out["encoder_out"][0].view(-1, hidden_state_size)
        num_samples = flatten_encoder_ctc_out.shape[0]
        encoded_ctc_classification_label = torch.zeros(num_samples, device=flatten_encoder_ctc_out.device, dtype=torch.long)
        encoded_rec_classification_label = torch.ones(num_samples, device=flatten_encoder_rec_out.device, dtype=torch.long)
        GAN_loss_ctc = self.cross_entropy_loss(flatten_encoder_ctc_out, encoded_ctc_classification_label)
        GAN_loss_rec = self.cross_entropy_loss(flatten_encoder_rec_out, encoded_rec_classification_label)
        GAN_loss = GAN_loss_ctc + GAN_loss_rec

        total_loss = ctc_loss + reconstruction_loss + GAN_loss

        ntokens = sample["ntokens"]

        sample_size = 1
        logging_output = {
            "loss": utils.item(total_loss.data),  # * sample['ntokens'],
            "ctc_loss": utils.item(ctc_loss.data),  # * sample['ntokens'],
            "reconstruction_loss": utils.item(reconstruction_loss.data),  # * sample['ntokens'],
            "gan_loss": utils.item(GAN_loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        return total_loss, sample_size, logging_output

    def get_ctc_loss(self, lprobs, sample):
        """
        Get the ctc loss of encoder output
        """

        input_lengths = sample["net_input"]["src_lengths"]
        pad_mask = (sample["target"] != self.pad_idx) & (sample["target"] != self.eos_idx) & (sample["target"]
                                                                                              != self.bos_idx)
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):

            if self.use_customized_loss:
                loss = custom_ctc_loss(lprobs.transpose(0, 1).float(),  # compatible with fp16
                                       sample["target"],
                                       input_lengths,
                                       target_lengths,
                                       blank=self.blank_idx,
                                       reduction="mean", )
                loss = torch.stack([a / b for a, b in zip(loss, target_lengths)])
                loss = torch.clip(loss, min=0, max=100)  # Clip the loss to avoid overflow
                loss = loss.mean()
            else:
                loss = F.ctc_loss(
                    lprobs.transpose(0, 1).float(),
                    targets_flat,
                    input_lengths,
                    target_lengths,
                    blank=self.blank_idx,
                    reduction="mean",  # We were using mean
                    zero_infinity=self.zero_infinity,
                )
            # change to customized loss to check whether we can solve the bug.
        if self.label_smoothing > 0:
            smoothed_loss = -lprobs.transpose(0, 1).mean(-1).mean()
            loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smoothed_loss

        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, **kwargs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        # if sample_size != ntokens:
        #     metrics.log_scalar(
        #         "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
        #     )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

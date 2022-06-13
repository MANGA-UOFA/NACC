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
import time
from customized_ctc import ctc_loss as custom_ctc_loss


@dataclass
class SummarizationCtcCriterionConfig(FairseqDataclass):
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
    short_output_penalty: float = field(
        default=0,
        metadata={
            "help": "the penalty to avoid generating epsilon to avoid short summary"
        }
    )
    use_customized_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to use customized ctc loss for CTC-training"
        }
    )


@register_criterion("summarization_ctc", dataclass=SummarizationCtcCriterionConfig)
class SummarizationCtcCriterion(FairseqCriterion):
    def __init__(self, cfg: SummarizationCtcCriterionConfig, task: FairseqTask) -> None:
        super().__init__(task)
        self.blank_idx = task.target_dictionary.blank()
        self.bos_idx = task.target_dictionary.bos()
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.vocab_size = len(task.target_dictionary)
        self.post_process = cfg.post_process

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg
        self.label_smoothing = cfg.label_smoothing
        self.use_customized_loss = cfg.use_customized_loss
        self.short_output_penalty = cfg.short_output_penalty
        self.cross_entropy = nn.CrossEntropyLoss()
        self.early_release_step = 5000
        self.step_count = 0

    def forward(self, model: BaseFairseqModel, sample: Dict, reduce=True):
        net_output = model(sample)
        self.step_count += 1
        lprobs = model.get_normalized_probs(net_output, log_probs=True).contiguous()  # (T, B, C) from the encoder

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

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
                    reduction="none",  # We were using mean
                    zero_infinity=self.zero_infinity,
                )
                if self.step_count > self.early_release_step:
                    loss = torch.clip(loss, min=0, max=100)  # Clip the loss to avoid overflow
                loss = loss.mean()
            # change to customized loss to check whether we can solve the bug.

        if self.short_output_penalty > 0:
            flattened_prob = lprobs.view(-1, self.vocab_size)
            epsilon_target = torch.zeros(len(flattened_prob), dtype=torch.long, device=lprobs.device)
            epsilon_target[:] = self.blank_idx
            # avoid generating epsilon
            penalty_loss = -self.cross_entropy(lprobs.view(-1, self.vocab_size), epsilon_target)

            loss += self.short_output_penalty * penalty_loss
        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = 1
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if self.label_smoothing > 0:
            smoothed_loss = -lprobs.transpose(0, 1).mean(-1).mean()
            loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smoothed_loss

        return loss, sample_size, logging_output

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

# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional, Dict, Union

import torch
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
    use_customized_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to use customized ctc loss for CTC-training"
        }
    )
    num_cross_layer_sample: int = field(
        default=0,
        metadata={
            "help": "Number of cross layer sample"
        }
    )


@register_criterion("summarization_ctc_sd", dataclass=SummarizationCtcCriterionConfig)
class SummarizationCtcCriterion(FairseqCriterion):
    def __init__(self, cfg: SummarizationCtcCriterionConfig, task: FairseqTask) -> None:
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
        self.num_cross_layer_sample = cfg.num_cross_layer_sample

    def sequence_ctc_loss_with_logits(self,
                                      logits: torch.FloatTensor,
                                      input_lengths,
                                      targets: torch.LongTensor,
                                      target_mask: Union[torch.FloatTensor, torch.BoolTensor],
                                      blank_index: torch.LongTensor,
                                      label_smoothing=0.0,
                                      reduce=True,
                                      max_loss=100,
                                      ) -> torch.FloatTensor:

        if len(targets.size()) == 1:
            targets = targets.unsqueeze(0)
            target_mask = target_mask.unsqueeze(0)
        target_lengths = (target_mask.bool()).long().sum(1)

        # (batch_size, T, n_class)
        log_probs = logits.log_softmax(-1)
        # log_probs_T : (T, batch_size, n_class), this kind of shape is required for ctc_loss
        log_probs_T = log_probs.transpose(0, 1)
        #     assert (target_lengths == 0).any()
        targets = targets.long()
        targets = targets[target_mask.bool()]
        if reduce:
            loss = F.ctc_loss(
                log_probs_T.float(),  # compatible with fp16
                targets,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="mean",
                zero_infinity=True,
            )
        else:
            loss = F.ctc_loss(
                log_probs_T.float(),  # compatible with fp16
                targets,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="none",
                zero_infinity=True,
            )
            loss = torch.stack([a / b for a, b in zip(loss, target_lengths)])
            loss = loss.clip(min=0, max=max_loss)
            loss = sum(loss)/len(loss)

        if label_smoothing > 0:
            raise NotImplementedError
            smoothed_loss = -log_probs.mean(-1).mean()
            loss = (1 - label_smoothing) * loss + label_smoothing * smoothed_loss
        return loss

    def forward(self, model: BaseFairseqModel, sample: Dict, reduce=False):
        output_logits_list = model(sample)

        if type(output_logits_list) != list:
            # During inference, the model only returns the last layer output.
            output_logits_list = [output_logits_list]

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

        with torch.backends.cudnn.flags(enabled=False):
            tgt_tokens = sample["target"]
            target_mask = tgt_tokens.ne(self.pad_idx)
            if self.num_cross_layer_sample != 0:
                output_logits_list = torch.stack(output_logits_list, dim=0)

                N_SAMPLE = self.num_cross_layer_sample

                num_decoder_layer = output_logits_list.size(0)
                num_tokens = tgt_tokens.size(1)
                num_vocab = output_logits_list.size(-1)
                batch_size = tgt_tokens.size(0)

                all_sample_ctc_loss = 0

                for sample_id in range(N_SAMPLE):
                    cross_layer_sampled_ids_ts = torch.randint(num_decoder_layer, (batch_size * num_tokens,),
                                                               device=output_logits_list.device)
                    output_logits_list = output_logits_list.view(num_decoder_layer, -1, num_vocab)
                    gather_idx = cross_layer_sampled_ids_ts.unsqueeze(1).expand(-1, num_vocab).unsqueeze(0)
                    gather_logits = output_logits_list.gather(0, gather_idx).view(batch_size, num_tokens, num_vocab)

                    target_mask = tgt_tokens.ne(self.pad)
                    # if self.args.use_ctc:
                    ctc_loss = self.sequence_ctc_loss_with_logits(
                        logits=gather_logits,
                        input_lengths=input_lengths,
                        targets=tgt_tokens,
                        target_mask=target_mask,
                        blank_index=self.blank_idx,
                        label_smoothing=self.label_smoothing,
                        reduce=reduce,
                    )
                    all_sample_ctc_loss += ctc_loss

                loss = all_sample_ctc_loss / len(output_logits_list)

            else:
                # if self.args.use_ctc:
                all_layer_ctc_loss = 0
                normalized_factor = 0
                for layer_idx, word_ins_out in enumerate(output_logits_list):
                    ctc_loss = self.sequence_ctc_loss_with_logits(
                        logits=word_ins_out,
                        input_lengths=input_lengths,
                        targets=tgt_tokens,
                        target_mask=target_mask,
                        blank_index=self.blank_idx,
                        label_smoothing=self.label_smoothing,  # NOTE: enable and double check with it later
                        reduce=reduce,
                    )
                    factor = 1  # math.sqrt(layer_idx + 1)
                    all_layer_ctc_loss += ctc_loss * factor
                    normalized_factor += factor
                loss = all_layer_ctc_loss / normalized_factor

        target_lengths = (target_mask.bool()).long().sum(1)
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

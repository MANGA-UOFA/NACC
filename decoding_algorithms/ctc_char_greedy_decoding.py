# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from decoding_algorithms.ctc_decoder_base import CTCDecoderBase
from typing import Dict, List
from fairseq.data.ctc_dictionary import CTCDictionary
from torch import TensorType
import torch


class CTCCharGreedyDecoder(CTCDecoderBase):
    """
    CTC greedy decoding decoder
    """

    def __init__(self, dictionary: CTCDictionary, decoder_parameters: Dict) -> None:
        super().__init__(dictionary, decoder_parameters)

    def decode(self, output_logits: TensorType, **kwargs) -> List[List[int]]:
        """
        Decoding function of the CTC greedy Decoder.
        """
        output_logits[output_logits == 0] -= 10 * torch.finfo(output_logits.dtype).eps
        if output_logits.dtype != torch.float16:
            output_logits = output_logits.cpu()  # Move everything to CPU for fair comparison (cpu decoding)
        scores, summary_list = output_logits.topk(1, -1)
        decoded_summary_list = []
        summary_list = summary_list.squeeze(-1).tolist()
        for summary in summary_list:
            processed_summary = self.ctc_post_processing(summary)
            if self.truncate:
                truncated_summary = []
                current_summary_char_length = 0
                for word in processed_summary:
                    next_step_length = len(self.dictionary[word]) + 1
                    if current_summary_char_length + next_step_length <= self.desired_length:
                        truncated_summary.append(word)
                        current_summary_char_length += next_step_length
                    else:
                        # use some very unlikely token to denote the last incomplete token.
                        truncated_summary.append(len(self.dictionary)-1)
                        break
                processed_summary = truncated_summary
            decoded_summary_list.append(processed_summary)
        return decoded_summary_list

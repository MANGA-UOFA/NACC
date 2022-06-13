# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import *
import torch
import numpy as np
from decoding_algorithms.ctc_decoder_base import CTCDecoderBase


class CTCBeamSearchCharLengthControlDecoder(CTCDecoderBase):
    """
    CTC beam search decoder with length control
    """

    def __init__(self, dictionary, decoder_parameters):
        super().__init__(dictionary, decoder_parameters)

        # Some temporal variables
        self.sample_desired_length = None
        self.id_to_index_dict_list = None
        self.prefix_lprob_table = None
        self.index_to_id_dict_list = None
        self.prefix_index_table = None
        self.ctc_sequence_length = None
        self.prob_sequence = None
        self.prev_max_row_index = None  # Maximum chars in the previous iteration.

        # Some decoder parameters
        self.force_length = decoder_parameters["force_length"]
        self.use_length_ratio = decoder_parameters["use_length_ratio"]
        self.k = decoder_parameters["k"]
        self.beam_size = decoder_parameters["beam_size"]
        self.blank_index = dictionary.blank()
        self.replacing_value = float("-inf")
        self.device = None
        self.dtype = None

        # Some assertions
        assert self.beam_size > 0, "Beam size are required to be positive"
        assert self.k > self.beam_size + 1, "The k has to be at least one above the beam size of the algorithm"
        assert self.desired_length > 0, "The desired length should be greater than 0"
        assert self.beam_size % 2 == 0, "The beam size must be even number"
        assert self.force_length == False, "There is no way to force the length for char-level length control"
        # Record the length of each word in the dictionary
        self.word_length_tensor = torch.zeros(len(dictionary), dtype=torch.long, device=self.device)
        for i in range(len(dictionary)):
            word = dictionary[i]
            word_len = len(word)
            # Set the length of special tokens to 0
            if word == dictionary.bos_word or word == dictionary.eos_word or \
                    word == dictionary.pad_word or word == dictionary.blank_word:
                word_len = 0
            self.word_length_tensor[i] = word_len

        self.half_beam_size = self.beam_size // 2
        self.half_beam_axis = torch.arange(0, self.half_beam_size, dtype=torch.long, device=self.device)
        self.beam_axis = torch.arange(0, self.beam_size, dtype=torch.long, device=self.device)
        self.special_element_tuple_list = [list(range(self.k)), [0] * self.k]

    def top_k_filtering(self, column, lprob):
        """
            Get the top-k most probable token and their corresponding probabilities
            logits (tensor): the logits returned by the model at the current time step

            Return:
                values (tensor of size k): the probability of the most probable tokens, with the first one set to be the blank token
                index_to_id_dict (dict of size k): a dictionary mapping from the element index of the values vector to their real id
                repeated_element_index_list (list): a list of the index (row, column) indicating the repeating index
        """
        top_k_id_tensor = torch.zeros(self.k, dtype=torch.long, device=self.device)  # word id
        top_k_lprob_tensor = torch.zeros(self.k, dtype=self.dtype, device=self.device)
        # Record the blank token id, no matter whether it's in top k
        top_k_id_tensor[0] = self.blank_index
        top_k_lprob_tensor[0] = lprob[self.blank_index]
        # Find the k most probable words and their indexes
        naive_top_k_lprob, naive_top_k_id = lprob.topk(self.k)
        # Fill in the remaining slot of the top_k_id_tensor and top_k_lprob_tensor
        top_k_id_tensor[1:] = naive_top_k_id[naive_top_k_id != self.blank_index][:self.k - 1]
        top_k_lprob_tensor[1:] = naive_top_k_lprob[naive_top_k_id != self.blank_index][:self.k - 1]

        # create dictionaries mapping between index and ids
        index_to_id_dict = {k: v.item() for k, v in enumerate(top_k_id_tensor)}
        id_to_index_dict = {v.item(): k for k, v in enumerate(top_k_id_tensor)}

        # Record the dictionary
        self.index_to_id_dict_list[column] = index_to_id_dict
        self.id_to_index_dict_list[column] = id_to_index_dict

        if column == 0:
            # For the first column, there is no repeated element
            repeated_or_special_element_index_list = self.special_element_tuple_list
        else:
            prev_id_to_index_dict = self.id_to_index_dict_list[column - 1]
            prev_index_to_id_dict = self.index_to_id_dict_list[column - 1]
            # Find the overlapping words except blank token in the current top_k words and previous top_k words
            repeated_element_list = set(prev_id_to_index_dict.keys()).intersection(set(top_k_id_tensor[1:].tolist()))
            repeated_element_tuple_index_list = [[prev_id_to_index_dict[element] for element in repeated_element_list],
                                                 [id_to_index_dict[element] for element in repeated_element_list]]
            repeated_or_special_element_index_list = repeated_element_tuple_index_list
            repeated_or_special_element_index_list[0] += self.special_element_tuple_list[0]
            repeated_or_special_element_index_list[1] += self.special_element_tuple_list[1]

        repeated_or_special_element_index_list[0] = tuple(repeated_or_special_element_index_list[0])
        repeated_or_special_element_index_list[1] = tuple(repeated_or_special_element_index_list[1])

        return top_k_lprob_tensor, repeated_or_special_element_index_list, top_k_id_tensor

    def get_blank_token_prob(self, current_filtered_log_prob):
        only_special_cloned_prob = current_filtered_log_prob.clone()
        only_special_cloned_prob[1:] = self.replacing_value
        return only_special_cloned_prob

    def get_non_blank_token_prob(self, current_filtered_log_prob):
        only_non_special_cloned_prob = current_filtered_log_prob.clone()
        only_non_special_cloned_prob[0] = self.replacing_value
        return only_non_special_cloned_prob

    def ctc_beam_search_length_control_initialization(self, logits):
        """
        Initialize some temporary variables
        """
        self.prob_sequence = logits.softmax(dim=-1)  # Get the log probability from logits
        self.ctc_sequence_length = len(self.prob_sequence)  # The length of the ctc output sequence.
        self.prefix_index_table = torch.zeros(
            [self.sample_desired_length, self.ctc_sequence_length, self.beam_size, self.ctc_sequence_length],
            dtype=torch.long, device=self.device) - 1
        # The main table to store the dictionary mapping from token index to token id
        self.index_to_id_dict_list = [-1] * self.ctc_sequence_length
        self.id_to_index_dict_list = [-1] * self.ctc_sequence_length
        lprob_table_dimensions = [self.sample_desired_length, self.ctc_sequence_length, self.beam_size]
        self.prefix_lprob_table = torch.zeros(lprob_table_dimensions, dtype=self.prob_sequence.dtype,
                                              device=self.device)
        self.prev_max_row_index = 0  # initialize max row index to 0

    def beam_search_row_inference(self, column, max_row_index, only_blank_cloned_prob, only_non_blank_cloned_prob,
                                  blank_or_repeated_transition_matrix, non_blank_non_repeated_transition_matrix,
                                  top_k_id_tensor):
        """
        Fill in each row of the dynamic programming table.
        """
        current_k_words_length = self.word_length_tensor[top_k_id_tensor]  # Length of the k chosen words.
        # Initialization of the first column
        if column == 0:
            # Assign the probability of blank token to the first beam
            self.prefix_lprob_table[0, column] += only_blank_cloned_prob[0:self.beam_size]
            self.prefix_index_table[0, column, 0, 0] = 0
            # Find the suitable starting words for each row
            first_word_lprob = only_non_blank_cloned_prob.repeat(max_row_index, 1)
            rows_allow_length = torch.arange(1, max_row_index + 1).unsqueeze(1)
            repeated_current_word_length = current_k_words_length.repeat(max_row_index, 1)
            allow_length_mask = repeated_current_word_length == rows_allow_length
            first_word_lprob[~allow_length_mask] = self.replacing_value
            top_beam_prob, top_beam_index = first_word_lprob.topk(self.beam_size, dim=1)
            # Record the starting words and their corresponding probability
            self.prefix_lprob_table[1:max_row_index+1, column, :] = top_beam_prob
            self.prefix_index_table[1:max_row_index+1, column, :, 0] = top_beam_index
            return  # we don't need any more operation for the first column
        else:
            # Given only first beam in the blank row the right prob
            self.prefix_lprob_table[0, column] = self.prefix_lprob_table[0, column - 1] + \
                                                 only_blank_cloned_prob[0:self.beam_size]
            # All tokens in the first row must be blank
            self.prefix_index_table[0, column, 0, 0:column + 1] = 0

        # For pure expansion row
        if max_row_index != self.prev_max_row_index:
            num_expansion_rows = max_row_index - self.prev_max_row_index  # Number of pure expansion rows.
            expansion_rows_axis = torch.arange(0, num_expansion_rows)
            # We first calculate the transition probability from each previous cell to the current (k) words
            diagonal_neighbour_lprob = self.prefix_lprob_table[:self.prev_max_row_index+1, column - 1]
            diagonal_neighbour_last_index = self.prefix_index_table[:self.prev_max_row_index+1, column - 1, :, column - 1]
            joint_diagonal_neighbour_lprob = \
                non_blank_non_repeated_transition_matrix[diagonal_neighbour_last_index.flatten()].view(self.prev_max_row_index + 1,
                                                                                                       self.beam_size,
                                                                                                       self.k
                                                                                                       ) \
                + diagonal_neighbour_lprob.unsqueeze(2)

            # Then we repeat this probability for each expansion row
            repeated_joint_diagonal_neighbour_lprob = joint_diagonal_neighbour_lprob.repeat(num_expansion_rows, 1, 1, 1)

            # Then we mask out invalid transitions (over-length sentences).
            current_rows_allow_length = torch.arange(1, num_expansion_rows + 1)[(..., ) + (None, ) * 3]
            prev_rows_allow_length = torch.arange(0, self.prev_max_row_index + 1).__reversed__()
            repeated_prev_rows_allow_length = \
                prev_rows_allow_length.\
                    repeat(num_expansion_rows * self.beam_size * self.k, 1).view(num_expansion_rows,
                                                                                 self.beam_size,
                                                                                 self.k,
                                                                                 self.prev_max_row_index + 1
                                                                                 ).transpose(1, 3).transpose(2, 3)
            rows_allow_length = current_rows_allow_length + repeated_prev_rows_allow_length
            repeated_current_word_length = \
                current_k_words_length.repeat(num_expansion_rows * (self.prev_max_row_index + 1) * self.beam_size, 1).\
                    view(num_expansion_rows,
                         self.prev_max_row_index + 1,
                         self.beam_size,
                         self.k
                         )
            allow_length_mask = (repeated_current_word_length <= rows_allow_length)
            repeated_joint_diagonal_neighbour_lprob[~allow_length_mask] = self.replacing_value

            # Finally, we find the most probable choice for each row
            reshaped_repeated_joint_diagonal_neighbour_lprob = \
                repeated_joint_diagonal_neighbour_lprob.view(num_expansion_rows,
                                                             (self.prev_max_row_index + 1) * self.beam_size,
                                                             self.k
                                                             )
            top_values_diagonal, top_index_diagonal = reshaped_repeated_joint_diagonal_neighbour_lprob.max(dim=1)
            # (num_expansion_rows, k)
            top_beam_value_diagonal, top_beam_index_diagonal = top_values_diagonal.topk(self.beam_size, dim=1)  # num_expansion_rows * beam_size
            top_beam_prev_index_diagonal = top_index_diagonal[expansion_rows_axis.unsqueeze(1), top_beam_index_diagonal]

            # Record the chosen index and the corresponding lprob
            self.prefix_lprob_table[self.prev_max_row_index + 1:max_row_index + 1, column] = top_beam_value_diagonal
            self.prefix_index_table[self.prev_max_row_index + 1:max_row_index + 1, column, :, 0:column] = \
                self.prefix_index_table[:self.prev_max_row_index + 1, column - 1, :, 0:column].\
                    reshape((self.prev_max_row_index + 1)*self.beam_size, -1)[top_beam_prev_index_diagonal.flatten()].\
                    view(num_expansion_rows, self.beam_size, -1)
            self.prefix_index_table[self.prev_max_row_index + 1:max_row_index + 1, column, :, column] = top_beam_index_diagonal

        if column > 0:  # If we are not at the first column, then there is middle rows.
            # get the prob & last index of all beams of all middle rows
            num_middle_rows = self.prev_max_row_index  # we subtract the first row
            diagonal_neighbour_last_index = self.prefix_index_table[0:self.prev_max_row_index, column - 1, :, column - 1]
            row_neighbour_last_index = self.prefix_index_table[1:self.prev_max_row_index + 1, column - 1, :, column - 1]
            diagonal_neighbour_lprob = self.prefix_lprob_table[0:self.prev_max_row_index, column - 1]
            row_neighbour_lprob = self.prefix_lprob_table[1:self.prev_max_row_index + 1, column - 1]

            # Add the log-prob of current step and previous time step
            joint_diagonal_neighbour_lprob = \
                non_blank_non_repeated_transition_matrix[diagonal_neighbour_last_index.flatten()].view(self.prev_max_row_index,
                                                                                                       self.beam_size,
                                                                                                       self.k
                                                                                                       ) \
                + diagonal_neighbour_lprob.unsqueeze(2)
            joint_row_neighbour_lprob = \
                blank_or_repeated_transition_matrix[row_neighbour_last_index.flatten()].view(num_middle_rows,
                                                                                             self.beam_size,
                                                                                             self.k) \
                + row_neighbour_lprob.unsqueeze(2)

            # Then we repeat diagonal probability for each expansion row
            repeated_joint_diagonal_neighbour_lprob = joint_diagonal_neighbour_lprob.repeat(num_middle_rows, 1, 1, 1)

            # Then we mask out invalid transitions (over-length sentences).
            current_rows_allow_length = torch.arange(1, num_middle_rows + 1)[(..., ) + (None, ) * 3]
            prev_rows_allow_length = torch.arange(0, self.prev_max_row_index)
            repeated_prev_rows_allow_length = \
                prev_rows_allow_length.\
                    repeat(num_middle_rows * self.beam_size * self.k, 1).view(num_middle_rows,
                                                                                 self.beam_size,
                                                                                 self.k,
                                                                                 self.prev_max_row_index
                                                                                 ).transpose(1, 3).transpose(2, 3)
            rows_allow_length = current_rows_allow_length - repeated_prev_rows_allow_length
            repeated_current_word_length = \
                current_k_words_length.repeat(num_middle_rows * (self.prev_max_row_index) * self.beam_size, 1).\
                    view(num_middle_rows,
                         self.prev_max_row_index,
                         self.beam_size,
                         self.k
                         )
            allow_length_mask = (repeated_current_word_length <= rows_allow_length)
            repeated_joint_diagonal_neighbour_lprob[~allow_length_mask] = self.replacing_value

            # Finally, we find the most probable choice for each row
            reshaped_repeated_joint_diagonal_neighbour_lprob = \
                repeated_joint_diagonal_neighbour_lprob.view(num_middle_rows,
                                                             self.prev_max_row_index * self.beam_size,
                                                             self.k
                                                             )
            top_values_diagonal, top_index_diagonal = reshaped_repeated_joint_diagonal_neighbour_lprob.max(dim=1)
            # (num_expansion_rows, k)
            top_beam_value_diagonal, top_beam_index_diagonal = top_values_diagonal.topk(self.half_beam_size, dim=1)  # num_expansion_rows * beam_size
            top_beam_prev_index_diagonal \
                = top_index_diagonal[torch.arange(0, num_middle_rows).unsqueeze(1), top_beam_index_diagonal]

            # Find the top beam index for row transitions.
            top_values_row, top_index_row = joint_row_neighbour_lprob.max(dim=1)
            _, top_beam_index_row = top_values_row.topk(self.half_beam_size)

            self.prefix_lprob_table[1:self.prev_max_row_index + 1, column, 0:self.half_beam_size] = top_beam_value_diagonal
            self.prefix_index_table[1:self.prev_max_row_index + 1, column, 0:self.half_beam_size, 0:column] = \
                self.prefix_index_table[:self.prev_max_row_index, column - 1, :, 0:column].\
                    reshape(self.prev_max_row_index*self.beam_size, -1)[top_beam_prev_index_diagonal.flatten()].\
                    view(num_middle_rows, self.half_beam_size, -1)
            self.prefix_index_table[1:self.prev_max_row_index + 1, column, 0:self.half_beam_size, column] = top_beam_index_diagonal

            axis_index = torch.arange(0, num_middle_rows).expand(self.half_beam_size, num_middle_rows).transpose(0, 1).flatten()
            formatted_top_beam_index_row = [axis_index, top_beam_index_row.flatten()]
            self.prefix_lprob_table[1:self.prev_max_row_index + 1, column, self.half_beam_size:] = \
                top_values_row[formatted_top_beam_index_row].view(num_middle_rows, -1)
            self.prefix_index_table[1:self.prev_max_row_index + 1, column, self.half_beam_size:, 0:column] = \
                self.prefix_index_table[axis_index + 1, column - 1, top_index_row[formatted_top_beam_index_row],
                0:column].view(num_middle_rows, self.half_beam_size, -1)
            self.prefix_index_table[1:self.prev_max_row_index + 1, column, self.half_beam_size:, column] = top_beam_index_row

    def beam_search_column_inference(self, column):
        """
        Perform table (prob table and prefix table) filling for a single column
        """
        current_filtered_lprob, repeated_or_special_element_index_list, top_k_id_tensor = \
            self.top_k_filtering(column, self.prob_sequence[column].log())

        current_max_word_length = self.word_length_tensor[top_k_id_tensor].max()
        max_row_index = min(self.prev_max_row_index + current_max_word_length, self.sample_desired_length - 1)

        only_blank_cloned_prob = self.get_blank_token_prob(current_filtered_lprob)
        only_non_blank_cloned_prob = self.get_non_blank_token_prob(current_filtered_lprob)

        non_blank_non_repeated_transition_matrix = current_filtered_lprob.expand(self.k, self.k).clone()
        non_blank_non_repeated_transition_matrix[repeated_or_special_element_index_list] = self.replacing_value

        repeated_or_blank_transition_mask = (non_blank_non_repeated_transition_matrix == self.replacing_value)
        blank_or_repeated_transition_matrix = current_filtered_lprob.expand(self.k, self.k).clone()
        blank_or_repeated_transition_matrix[~repeated_or_blank_transition_mask] = self.replacing_value

        self.beam_search_row_inference(column, max_row_index, only_blank_cloned_prob, only_non_blank_cloned_prob,
                                       blank_or_repeated_transition_matrix, non_blank_non_repeated_transition_matrix,
                                       top_k_id_tensor)
        self.prev_max_row_index = max_row_index

    def determine_whether_length_control_needed(self, logits, source_length):
        """
        Determines whether length control is needed.
        """
        # Determine the desired summary length
        if self.use_length_ratio:
            # If the desired length is proportional to the input length, set desired length based on the input
            self.sample_desired_length = int(np.floor(0.01 * self.desired_length * source_length))
        else:
            # Otherwise, use the given length
            self.sample_desired_length = self.desired_length + 1  # Handle 0-length summary

        # Check the whether greedy decoding gives shorter summary
        _, greedy_summary_index = logits.max(-1)
        greedy_summary = self.ctc_post_processing(greedy_summary_index.tolist())
        # Determine whether we should adopt greedy decoding for this summary generation
        decoded_greedy_summary_length = sum([self.word_length_tensor[i].item() for i in greedy_summary])
        use_shorter_summary = (decoded_greedy_summary_length <= self.sample_desired_length - 1) and (not self.force_length)
        if use_shorter_summary:
            return False, greedy_summary
        else:
            return True, greedy_summary

    def ctc_beam_search_char_length_control(self, logits, source_length):
        """
        This is the main script for ctc beam search length control decoding. Currently, this script does not support
        parallel decoding.
        """
        # First check whether length control is needed
        need_length_control, greedy_summary = self.determine_whether_length_control_needed(logits, source_length)
        if not need_length_control:
            # If not needed, return greedily decoded summary.
            return greedy_summary

        # Initialization
        self.ctc_beam_search_length_control_initialization(logits)

        # Main Loop
        for column in range(0, self.ctc_sequence_length):
            # For each column, we calculate the desired probability and the prefix for each allowable table slot.
            self.beam_search_column_inference(column)

        max_path_index = torch.argmax(self.prefix_lprob_table[-1, -1])
        path = self.prefix_index_table[self.sample_desired_length - 1, self.ctc_sequence_length - 1, max_path_index]
        generated_summary = []
        for i in range(0, self.ctc_sequence_length):
            current_dict = self.index_to_id_dict_list[i]
            current_token = current_dict[path[i].item()]
            generated_summary.append(current_token)

        generated_summary = self.ctc_post_processing(generated_summary)
        generated_summary_char_length = sum([self.word_length_tensor[i] for i in generated_summary])
        assert generated_summary_char_length <= self.sample_desired_length - 1, "Generated summary has a wrong length"
        return generated_summary

    def decode(self, output_logits, source_length):
        if output_logits.dtype != torch.float16:
            output_logits = output_logits.cpu()  # Move everything to CPU for fair comparison (cpu decoding)
        self.dtype = output_logits.dtype
        self.device = output_logits.device
        decoded_summary_list = []
        for i in range(0, len(output_logits)):
            decoded_summary = self.ctc_beam_search_char_length_control(output_logits[i], self.desired_length)
            decoded_summary_list.append(decoded_summary)
        return decoded_summary_list


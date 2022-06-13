# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import numpy as np
from decoding_algorithms.ctc_decoder_base import CTCDecoderBase


class CTCScopeSearchCharLengthControlDecoder(CTCDecoderBase):
    """
    An equivalent implementation of the CTC Beam Search with Length Control (BSLC).
    Comparing with naive BSLC, this implementation can generalize toward brute force search by increasing the scope.
    """

    def __init__(self, dictionary, decoder_parameters):
        super().__init__(dictionary, decoder_parameters)
        # Sample temporary variable
        self.sample_desired_length = None  # Desired length for the specific sample
        self.id_to_index_dict_list = None
        self.index_to_id_dict_list = None
        self.ctc_sequence_length = None
        self.top_k_lprob_sequence = None
        self.top_k_id_sequence = None
        self.prev_max_row_index = None
        self.scope_lprob_table = None
        self.transition_tracker_table = None
        self.single_prefix_index_table = None
        # self.transition_length_limit_table = None
        self.scale_axis = None  # [0, ..., self.desired_length // self.scaling_factor]
        self.scope_length_table = None  # track the summary length of each choice
        self.prev_chosen_index_list = None
        self.prev_chosen_index = None  # Record the (marginalized) previous chosen index
        self.device = None
        self.dtype = None

        # Decoder configuration parameters
        self.force_length = decoder_parameters["force_length"]
        self.use_length_ratio = decoder_parameters["use_length_ratio"]
        self.k = decoder_parameters["k"]  # dummy variable
        self.beam_size = decoder_parameters["beam_size"]
        self.scope = decoder_parameters["scope"]
        self.scaling_factor = decoder_parameters["scaling_factor"]
        self.margin_criteria = decoder_parameters["marg_criteria"]
        self.blank_id = dictionary.blank()
        self.replacing_value = float("-inf")
        self.explict_length = True

        # Assertions on decoder parameters
        assert self.scope > 1, "The scope must be positive integer"
        assert self.beam_size > 0, "Beam size are required to be positive"
        assert self.desired_length > 0, "The desired length should be greater than 0"

        # Initialize reusable variables
        self.special_element_tuple_list = [list(range(self.k)), [0] * self.k]
        # Record the length of each word in the dictionary
        self.word_length_tensor = torch.zeros(len(dictionary), dtype=torch.long)
        self.masked_word_length_tensor = torch.zeros(len(dictionary), dtype=torch.long)  # special word has length 127
        special_word_list = [dictionary.bos_word, dictionary.eos_word, dictionary.pad_word, dictionary.blank_word]
        for i in range(len(dictionary)):
            word = dictionary[i]
            word_len = len(word)
            mask_word_len = len(word)
            # Set the length of special tokens to 0
            if word in special_word_list:
                word_len = 0
                mask_word_len = 127  # Avoid these word to be chosen in normal transitions. (lower bound violation)
            self.word_length_tensor[i] = word_len
            self.masked_word_length_tensor[i] = mask_word_len

        self.beam_axis = torch.arange(0, self.beam_size, dtype=torch.long, device=self.device)
        self.special_element_tuple_list = [list(range(self.k)), [0] * self.k]
        self.index_tensor = torch.arange(0, self.k, dtype=torch.long, device=self.device)

    def top_k_filtering(self, column):
        """
            Get the top-k most probable token and their corresponding probabilities
            logits (tensor): the logits returned by the model at the current time step

            Return:
                values (tensor of size k): the probability of the most probable tokens, with the first one set to be the blank token
                index_to_id_dict (dict of size k): a dictionary mapping from the element index of the values vector to their real id
                repeated_element_index_list (list): a list of the index (row, column) indicating the repeating index
        """
        top_k_lprob_tensor, top_k_id_tensor = self.top_k_lprob_sequence[column], self.top_k_id_sequence[column]

        # create dictionaries mapping between index and ids
        top_k_id_tensor_list = top_k_id_tensor.tolist()
        index_to_id_dict = {k: v for k, v in enumerate(top_k_id_tensor_list)}
        id_to_index_dict = {v: k for k, v in enumerate(top_k_id_tensor_list)}

        # Record the dictionary
        self.index_to_id_dict_list[column] = index_to_id_dict
        self.id_to_index_dict_list[column] = id_to_index_dict
        # "Convert" the dictionary to tensor for faster indexing

        if column == 0:
            # For the first column, there is no repeated element
            repeated_or_special_element_index_list = self.special_element_tuple_list
        else:
            prev_id_to_index_dict = self.id_to_index_dict_list[column - 1]
            # Find the overlapping words except blank token in the current top_k words and previous top_k words
            prev_id_set = set(prev_id_to_index_dict.keys())
            current_id_set = set(top_k_id_tensor_list[1:])
            repeated_element_list = prev_id_set.intersection(current_id_set)
            repeated_element_tuple_index_list = [[prev_id_to_index_dict[key] for key in repeated_element_list],
                                                 [id_to_index_dict[key] for key in repeated_element_list]]
            repeated_or_special_element_index_list = repeated_element_tuple_index_list
            repeated_or_special_element_index_list[0] += self.special_element_tuple_list[0]
            repeated_or_special_element_index_list[1] += self.special_element_tuple_list[1]

        repeated_or_special_element_index_list[0] = tuple(repeated_or_special_element_index_list[0])
        repeated_or_special_element_index_list[1] = tuple(repeated_or_special_element_index_list[1])

        return top_k_lprob_tensor, repeated_or_special_element_index_list, top_k_id_tensor

    def scope_search_row_inference(self, column, current_max_row_index, reshaping_index, filtered_lprob_matrix,
                                   blank_or_repeated_transition_matrix, non_blank_non_repeated_transition_matrix,
                                   current_k_words_length, current_mask_k_words_length):
        """
        Perform actual table filling for all rows in the given column
        """
        # Initialization
        if column == 0:
            assert current_k_words_length.max() <= self.sample_desired_length, \
                "Desired length must be larger than the maximum length of dictionary words"
            # If blank token is chosen, the next chosen word is going to be the leading word, which don't require
            # an extra space.
            current_k_words_length[0] = -1
            length_mask = self.find_length_mask(column, current_max_row_index, current_k_words_length)
            length_mask[0, 0] = True  # <blank> transition in the first row is allowed
            filtered_lprob_matrix[~length_mask] = float("-inf")  # Mask out non-allowable fillings
            top_values, top_indices = filtered_lprob_matrix.topk(self.beam_size, dim=1)  # top fillings for each row
            self.scope_lprob_table[:current_max_row_index + 1] += top_values[reshaping_index]
            self.scope_length_table[:current_max_row_index + 1] += current_k_words_length[top_indices][reshaping_index]
            self.prev_chosen_index_list[column] = top_indices
            self.prev_chosen_index = top_indices
            # No other operation is needed for the first column
            return
        # Recursion
        # Create some temporal variables for later usage
        blank_or_repeated_transition_matrix = blank_or_repeated_transition_matrix[self.prev_chosen_index]
        non_blank_non_repeated_transition_matrix = non_blank_non_repeated_transition_matrix[self.prev_chosen_index]
        num_dummy_dim = len(reshaping_index)
        prev_row_axis = [*range(self.prev_max_row_index + 1)]
        # Mask for non-blank-non-repeating transitions
        length_mask = self.find_length_mask(column, current_max_row_index, current_mask_k_words_length, num_dummy_dim,
                                            reshaping_index)
        reshaped_prev_lprob = self.scope_lprob_table[tuple([...] + [0]*num_dummy_dim)][(...,) + (None,) * num_dummy_dim]

        # Add a dim to adapt current rows
        reshaped_prev_lprob = reshaped_prev_lprob[0:self.prev_max_row_index + 1].unsqueeze(1)
        non_blank_non_repeated_transition_matrix = non_blank_non_repeated_transition_matrix.unsqueeze(1)
        # Blank/repeat transition probability and non-blank-non-repeat transition probability
        row_neighbouring_prob = reshaped_prev_lprob[:, 0] + blank_or_repeated_transition_matrix[reshaping_index]
        diagonal_neighbouring_prob = reshaped_prev_lprob + non_blank_non_repeated_transition_matrix[reshaping_index]
        diagonal_neighbouring_prob = diagonal_neighbouring_prob.repeat([1] + [current_max_row_index+1] + self.scope*[1])
        diagonal_neighbouring_prob[~length_mask] = self.replacing_value  # Mask out non-allowed transitions
        diagonal_neighbouring_prob[(prev_row_axis, prev_row_axis)] = row_neighbouring_prob  # Merge the transition prob

        # Transpose some dimensions, then view the joint prob as (self.beam_size * prev_rows)
        diagonal_neighbouring_prob = diagonal_neighbouring_prob.transpose(1, -1 * num_dummy_dim).contiguous()
        first_two_dim = diagonal_neighbouring_prob.shape[0:2]
        new_view_dim = [-1] + list(diagonal_neighbouring_prob.shape[2:])
        diagonal_neighbouring_prob = diagonal_neighbouring_prob.view(new_view_dim)

        # Take the top beam for each row.
        top_beam_values, top_beam_index = diagonal_neighbouring_prob.topk(self.beam_size, dim=0)
        # Find the row transition history of the selected beams
        # TODO: Find a solution to avoid using numpy to improve efficiency
        top_beam_index_numpy = top_beam_index.unsqueeze(-1).cpu().numpy()
        unraveled_index = np.unravel_index(top_beam_index_numpy, first_two_dim)
        unraveled_index = np.concatenate(unraveled_index, axis=-1)
        unraveled_index = torch.from_numpy(unraveled_index).to(self.device)
        # unraveled_index = self.unravel_indices(top_beam_index, first_two_dim)
        top_beam_values = top_beam_values.transpose(0, -1 * num_dummy_dim)
        top_row_index = unraveled_index[..., 0].transpose(0, -1 * num_dummy_dim).contiguous()
        top_word_index = unraveled_index[..., 1].transpose(0, -1 * num_dummy_dim).contiguous()
        flatten_row_index = top_row_index.view(current_max_row_index + 1, -1)

        # Find the history rows
        row_history_reshaping_index = tuple([...] + [0, None]*(num_dummy_dim-1))
        length_history_reshaping_index = tuple([...] + [0, None]*(num_dummy_dim-1))
        # TODO: torch.movedim operation increases the runtime, find a solution ot avoid using torch.movedim
        reshaped_transition_tracker_table = self.transition_tracker_table[:self.prev_max_row_index + 1].movedim(-1, 0)
        row_history = reshaped_transition_tracker_table[row_history_reshaping_index].movedim(0, -1)
        length_history = self.scope_length_table[:self.prev_max_row_index + 1][length_history_reshaping_index]
        flatten_row_history = row_history.view(self.prev_max_row_index + 1, -1, self.ctc_sequence_length)
        flatten_length_history = length_history.view(self.prev_max_row_index + 1, -1)
        transition_row_index = [flatten_row_index, torch.arange(0, flatten_row_index.shape[-1], device=self.device)]
        transition_length_index = [flatten_row_index, torch.arange(0, flatten_row_index.shape[-1], device=self.device)]
        new_row = flatten_row_history[transition_row_index]
        new_length = flatten_length_history[transition_length_index]

        # Record the current row
        new_row[..., column-1] = flatten_row_index
        # Calculate the current length
        blank_or_repeated_transition_mask = blank_or_repeated_transition_matrix != float("-inf")
        blank_or_repeated_index = blank_or_repeated_transition_mask.nonzero(as_tuple=True)
        partial_word_index = top_word_index[blank_or_repeated_index[:-1]]
        partial_word_index[partial_word_index == blank_or_repeated_index[-1].unsqueeze(1)[reshaping_index]] = 0
        top_word_index[blank_or_repeated_index[:-1]] = partial_word_index
        current_k_words_length += 1  # Account for the space
        current_k_words_length[0] = 0  # Repeated word and blank word don't need a space since they will be dropped.
        current_chosen_word_length = current_k_words_length[top_word_index]
        new_length += current_chosen_word_length.view(current_max_row_index + 1, -1)

        # Record
        correct_row_shape = list(row_history.shape)
        correct_row_shape[0] = current_max_row_index + 1
        correct_length_shape = self.scope_length_table[:current_max_row_index + 1][length_history_reshaping_index].shape
        new_row=new_row.view(correct_row_shape).expand(self.transition_tracker_table[:current_max_row_index+1].shape)
        new_length=new_length.view(correct_length_shape).expand(self.scope_length_table[:current_max_row_index+1].shape)
        self.transition_tracker_table[:current_max_row_index + 1] = new_row
        self.scope_length_table[:current_max_row_index + 1] = new_length
        self.scope_lprob_table[:current_max_row_index + 1] = top_beam_values.expand([-1] + [self.beam_size]*self.scope)
        self.prev_chosen_index_list[column] = top_word_index[[...] + [0] * (num_dummy_dim - 1)]
        self.prev_chosen_index = top_word_index[[...] + [0] * (num_dummy_dim - 1)]

    def scope_search_column_inference(self, column):
        """
        Perform table (prob table and prefix table) filling for a single column
        """
        # Calculate some temporal variables
        remaining_scope = self.scope - column  # Remaining dimensions before filling in the current column
        filtered_lprob, repeated_or_special_element_index_list, top_k_id_tensor = self.top_k_filtering(column)
        # Find the maximum row index of the current iteration
        current_k_words_length = self.word_length_tensor[top_k_id_tensor]
        current_mask_k_words_length = self.masked_word_length_tensor[top_k_id_tensor]
        current_max_word_length = current_k_words_length.max()
        current_max_total_length = self.scope_length_table.max() + current_max_word_length + 1
        current_max_row_index = min(math.ceil(current_max_total_length / self.scaling_factor), len(self.scale_axis) - 1)
        current_max_row_index = max(current_max_row_index, self.prev_max_row_index)
        # Create a mask for non_blank_non_repeat_transitions
        repeated_or_blank_transition_mask = torch.zeros([self.k, self.k], dtype=torch.bool, device=self.device)
        repeated_or_blank_transition_mask[repeated_or_special_element_index_list] = True
        # Mask out the blank and repeated transitions
        non_blank_non_repeated_transition_matrix = filtered_lprob.expand(self.k, self.k).clone()
        non_blank_non_repeated_transition_matrix[repeated_or_blank_transition_mask] = self.replacing_value
        # Mask out the non-blank and non-repeated transitions
        blank_or_repeated_transition_matrix = filtered_lprob.expand(self.k, self.k).clone()
        blank_or_repeated_transition_matrix[~repeated_or_blank_transition_mask] = self.replacing_value

        # Find the appropriate reshaping index and marginalize the lprob matrix if needed.
        if remaining_scope > 0:
            reshaping_index = tuple((...,) + (None,) * (remaining_scope - 1))
        else:
            most_probable_word_index = self.margin_over_prob_table()
            reshaping_index = tuple((...,) + (None,) * (1 - 1))
            self.single_prefix_index_table[0:self.prev_max_row_index + 1, column - 1] = most_probable_word_index

        # Call the row processing function
        filtered_lprob_matrix = filtered_lprob.unsqueeze(0).repeat(current_max_row_index + 1, 1)
        self.scope_search_row_inference(column, current_max_row_index, reshaping_index, filtered_lprob_matrix,
                                        blank_or_repeated_transition_matrix, non_blank_non_repeated_transition_matrix,
                                        current_k_words_length, current_mask_k_words_length)
        # record
        self.prev_max_row_index = current_max_row_index

    def margin_over_prob_table(self):
        """
        Marginalize over the first dimension of the table
        """
        remaining_axis = tuple(range(2, self.scope + 1))  # A tuple of the remaining axis after marginalization
        if self.margin_criteria == "mean":
            sum_old_prob_along_remaining_axis = torch.logsumexp(self.scope_lprob_table[:self.prev_max_row_index + 1],
                                                                dim=remaining_axis)
            most_probable_word_index = torch.argmax(sum_old_prob_along_remaining_axis, dim=1)
        elif self.margin_criteria == "filtered_mean":
            # Select token based on its average non-inf probability
            sum_lprob_along_remaining_axis \
                = torch.logsumexp(self.scope_lprob_table[:self.prev_max_row_index + 1], dim=remaining_axis)
            non_inf_sum = (self.scope_lprob_table[:self.prev_max_row_index + 1] != float("-inf")).long().sum(
                remaining_axis)
            sum_lprob_along_remaining_axis -= non_inf_sum.log()  # take the average
            sum_lprob_along_remaining_axis = torch.nan_to_num(sum_lprob_along_remaining_axis, float("-inf"))
            most_probable_word_index = torch.argmax(sum_lprob_along_remaining_axis, dim=1)
        elif self.margin_criteria == "max":
            # If we are using max as the select criteria, we select the token in the first axis that can lead to the
            # sub-sequence with the maximum probability
            max_old_prob_along_remaining_axis = self.scope_lprob_table[:self.prev_max_row_index + 1].\
                amax(dim=remaining_axis)
            most_probable_word_index = torch.argmax(max_old_prob_along_remaining_axis, dim=1)
        else:
            raise NotImplementedError("Haven't designed other evaluation criteria")

        # Marginalize the lprob scope table based on the chosen words.
        repeat_index = tuple([-1] + (self.scope - 1) * [-1] + [self.beam_size])
        row_axis = torch.arange(0, self.prev_max_row_index + 1)
        self.scope_lprob_table[0:self.prev_max_row_index + 1] \
            = self.scope_lprob_table[row_axis, most_probable_word_index].unsqueeze(-1).expand(repeat_index)
        # Marginalize the length table based on the chosen words.
        self.scope_length_table[0:self.prev_max_row_index + 1] \
            = self.scope_length_table[row_axis, most_probable_word_index].unsqueeze(-1).expand(repeat_index)
        # Marginalize the prev_chosen_index, notice this index is between 0...beam_size - 1
        self.prev_chosen_index = self.prev_chosen_index[row_axis, most_probable_word_index]
        # Marginalize the transition scope table based on the chosen words.
        repeat_index = tuple([-1] + (self.scope - 1) * [-1] + [self.beam_size] + [-1])
        self.transition_tracker_table[0:self.prev_max_row_index + 1] \
            = self.transition_tracker_table[row_axis, most_probable_word_index].unsqueeze(-2).expand(repeat_index)

        return most_probable_word_index

    def find_length_mask(self, column, current_max_row_index, current_k_words_length, num_dummy_dim=None,
                         reshaping_index=None):
        """
        Find a mask of transitions/fillings that satisfy the length requirement of the table.
        """
        if column == 0:
            # The first column requires fillings
            # Notice we include a dummy row in the lower bound
            length_lower_bound = torch.zeros(current_max_row_index + 1, dtype=torch.long, device=self.device)
            length_lower_bound[1:] = self.scale_axis[:current_max_row_index]
            length_lower_bound = length_lower_bound.unsqueeze(1).expand(-1, self.k)  # Expand it to a row * k matrix
            length_upper_bound = self.scale_axis[:current_max_row_index + 1].unsqueeze(1).expand(-1, self.k)
            # Return the length mask
            return (length_lower_bound < current_k_words_length) & (current_k_words_length <= length_upper_bound)
        else:
            # Move things to GPU for faster calculation
            # TODO: Decide whether to bound total length by scale axis or bound incremental length by transition matrix
            # Find the summary length of all word & prefix combinations
            prev_length = self.scope_length_table[:self.prev_max_row_index + 1][
                tuple([...] + [0]*num_dummy_dim)][(...,) + (None,) * num_dummy_dim]
            # Create an extra dimension for current rows
            reshaped_prev_length = prev_length.unsqueeze(1)
            # Other columns needs transitions
            if self.explict_length:
                # Notice we include a dummy row in the lower bound
                length_lower_bound = torch.zeros([current_max_row_index+1], dtype=torch.long, device=self.device)
                length_lower_bound[1:] = self.scale_axis[:current_max_row_index]
                length_lower_bound = length_lower_bound[(...,) + (None,) * self.scope]
                length_upper_bound = self.scale_axis[:current_max_row_index + 1][(...,) + (None,) * self.scope]
                current_k_words_length += 1  # account for the space
            else:
                # Notice we include a dummy row in the lower bound
                length_lower_bound = torch.zeros([self.prev_max_row_index+1, current_max_row_index+1],
                                                 dtype=torch.long, device=self.device)
                # Recalling
                length_lower_bound = length_lower_bound[(...,) + (None,) * self.scope]
                length_lower_bound[:, 1:] = \
                    self.transition_length_limit_table[:self.prev_max_row_index + 1, :current_max_row_index]
                length_upper_bound = \
                    self.transition_length_limit_table[:self.prev_max_row_index + 1, :current_max_row_index + 1]
                reshaped_prev_length[...] = 0  # We only care about incremental length change in this case

            total_length = reshaped_prev_length + current_k_words_length[reshaping_index]
            # Return the length mask
            length_mask = torch.logical_and(total_length > length_lower_bound, total_length <= length_upper_bound)
            return length_mask

    def ctc_scope_search_length_control_initialization(self, logits):
        """
        Initialize some temporary variables
        """
        lprob_sequence = logits
        naive_top_k_lprob_sequence, naive_top_k_id_sequence = lprob_sequence.topk(self.k, dim=-1)
        self.ctc_sequence_length = len(lprob_sequence)  # The length of the ctc output sequence.
        self.top_k_lprob_sequence = torch.zeros([self.ctc_sequence_length,self.k], dtype=self.dtype, device=self.device)
        self.top_k_id_sequence = torch.zeros([self.ctc_sequence_length, self.k], dtype=torch.long, device=self.device)
        self.top_k_lprob_sequence[:, 0] = lprob_sequence[:, self.blank_id]
        self.top_k_id_sequence[:, 0] = self.blank_id
        # Fill in the remaining slot of the top_k_id_tensor and top_k_lprob_tensor
        naive_top_k_lprob_sequence[naive_top_k_id_sequence == self.blank_id] = self.replacing_value  # Mask out blank
        _, new_index = naive_top_k_lprob_sequence.sort(descending=True, dim=-1)
        temp_axis = torch.arange(0, self.ctc_sequence_length, device=self.device).unsqueeze(1)
        self.top_k_id_sequence[:, 1:] = naive_top_k_id_sequence[temp_axis, new_index[:, :self.k - 1]]
        self.top_k_lprob_sequence[:, 1:] = naive_top_k_lprob_sequence[temp_axis, new_index[:, :self.k - 1]]

        assert self.scope < self.ctc_sequence_length, \
            "The scope to reduce cannot exceed the length of the ctc output sequence"
        # This assertion results from our int8 declaration of transition tracking tensor
        assert self.ctc_sequence_length <= 2 ** 7, "The sequence length cannot exceed 128"
        # Initialize a table to record the maximum possible length transitioning between rows.
        self.initialize_length_table()
        # Track the probability of all transitions within a scope
        scope_lprob_dimensions = [len(self.scale_axis)] + self.scope * [self.beam_size]
        self.scope_lprob_table = torch.zeros(scope_lprob_dimensions, dtype=self.dtype, device=self.device)
        self.scope_length_table = torch.zeros(scope_lprob_dimensions, dtype=torch.long, device=self.device)
        # Track the parent rows of all choices
        transition_tracker_dimensions = [len(self.scale_axis)]+self.scope*[self.beam_size]+[self.ctc_sequence_length]
        self.transition_tracker_table = torch.zeros(transition_tracker_dimensions, dtype=torch.long, device=self.device)
        self.transition_tracker_table[:] = -1
        # This table stores the i-(scope-1) th prefix at the i-th column.
        self.single_prefix_index_table = \
            torch.zeros([len(self.scale_axis), self.ctc_sequence_length], dtype=torch.long, device=self.device) - 1
        # Initialize a list of dictionary to record the mapping between index and word id at different time step.
        self.index_to_id_dict_list = [-1] * self.ctc_sequence_length
        self.id_to_index_dict_list = [-1] * self.ctc_sequence_length
        self.prev_chosen_index_list = [-1] * self.ctc_sequence_length
        # Initialize
        self.prev_max_row_index = 0
        self.prev_chosen_index = None
        self.word_length_tensor = self.word_length_tensor.to(self.device)
        self.masked_word_length_tensor = self.masked_word_length_tensor.to(self.device)

    def initialize_length_table(self):
        """
        Create a table to record the maximum allowable length for transitions between different table rows.
        """
        if self.scaling_factor == 1:
            row_axis = torch.arange(0, self.sample_desired_length+1, dtype=torch.long, device=self.device)
        else:
            row_axis = torch.arange(0, self.sample_desired_length, dtype=torch.long, device=self.device)

        self.transition_length_limit_table = row_axis.unsqueeze(0) - row_axis.unsqueeze(1)
        self.transition_length_limit_table -= 1  # -1 to accommodate the space between words
        self.transition_length_limit_table = self.transition_length_limit_table[(..., ) + (None, ) * self.scope]
        # Notice we have an additional row (0-th row) to store the <blank> token in both cases
        # 0-th row is only for <blank>, 1-th row is for summary with length 0~scale, 2-th row is for scale~2*scale...
        self.scale_axis = torch.arange(0, self.sample_desired_length // self.scaling_factor + 1, device=self.device)
        self.scale_axis *= self.scaling_factor
        if self.scaling_factor == 1:
            self.scale_axis[-1] = self.sample_desired_length 
        else:
            self.scale_axis[-1] = self.sample_desired_length -1
        self.transition_length_limit_table \
            = self.transition_length_limit_table[self.scale_axis.unsqueeze(1), self.scale_axis]

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
        # Add 1 to count for leading space of each word, -1 to count the zero leading space of the first word
        decoded_greedy_summary_length = sum([self.word_length_tensor[i].item() + 1 for i in greedy_summary]) - 1
        use_shorter_summary = \
            (decoded_greedy_summary_length <= self.sample_desired_length - 1) and (not self.force_length)
        too_short_and_will_cause_error = len(logits) <= self.sample_desired_length // 8  # 8 is a heuristic number
        if use_shorter_summary or too_short_and_will_cause_error:
            return False, greedy_summary
        else:
            return True, greedy_summary

    def finalize_generation(self):
        """
        Map the table into a summary (in the form of a list of word index)
        """
        # The last element in this table should remain untouched (-1)
        assert self.single_prefix_index_table[-1, -1] == -1, "Something is not correct for the table generation"
        # We find the path of the last row in the table and decode the path into a summary.
        # Fetch the trajectory
        maximum_trajectory_index = self.scope_lprob_table[-1].view(-1).argmax()
        # Reformat the index
        maximum_trajectory_index = self.unravel_indices(maximum_trajectory_index, self.scope_lprob_table[-1].shape)
        # Fetch the history rows
        maximum_trajectory_row_index = self.transition_tracker_table[-1][tuple(maximum_trajectory_index)]
        # Map the trajectory to summary prefix
        summary_words_internal_index_list = []  # Internal index, [0, 6)
        summary_words_external_index_list = []  # External index, [0, 10), can be used to map back to the vocabulary
        for i in range(self.scope - 1, self.ctc_sequence_length - 1):
            # For i-th column
            row_index_i_th_column = maximum_trajectory_row_index[i]  # selected row in the i-th column of the trajectory
            i_th_internal_summary_index = self.single_prefix_index_table[row_index_i_th_column, i]  # i internal summary
            summary_words_internal_index_list.append(i_th_internal_summary_index.item())
            # Find the external index by tracing back to the row where internal index are chosen
            row_index_word_chosen = maximum_trajectory_row_index[i - self.scope + 1]
            i_th_external_summary_index = self.prev_chosen_index_list[i - self.scope + 1][
                row_index_word_chosen][tuple(summary_words_internal_index_list[-self.scope:])]
            summary_words_external_index_list.append(i_th_external_summary_index.item())

        # Find the chosen word of the last few time step
        maximum_trajectory_row_index[-1] = len(self.scope_lprob_table) - 1  # last row index
        for i in range(self.ctc_sequence_length - self.scope, self.ctc_sequence_length):
            row_index_i_th_column = maximum_trajectory_row_index[i]  # selected row in the i-th column of the trajectory
            i_th_internal_summary_index = maximum_trajectory_index[-1 * (self.ctc_sequence_length - i)]
            summary_words_internal_index_list.append(i_th_internal_summary_index.item())
            i_th_external_summary_index = self.prev_chosen_index_list[i][
                row_index_i_th_column][tuple(summary_words_internal_index_list[-self.scope:])]
            summary_words_external_index_list.append(i_th_external_summary_index.item())

        # Decode the index to word id
        generated_summary = []
        for i in range(0, self.ctc_sequence_length):
            current_token_index = summary_words_external_index_list[i]
            current_column_dict = self.index_to_id_dict_list[i]
            current_token_id = current_column_dict[current_token_index]
            generated_summary.append(current_token_id)
        return generated_summary

    def ctc_scope_search_char_length_control(self, logits, source_length):
        """
        This function perform length control on the output CTC logits of the model decoder.
        """
        # First check whether length control is needed
        need_length_control, greedy_summary = self.determine_whether_length_control_needed(logits, source_length)
        if not need_length_control:
            # If not needed, return greedily decoded summary.
            return greedy_summary
        # Initialization of temporal variables
        self.ctc_scope_search_length_control_initialization(logits)
        # Main Loop
        for column in range(0, self.ctc_sequence_length):
            self.scope_search_column_inference(column)
        # Finalize a generation based on previously calculated variables
        naive_generated_summary = self.finalize_generation()
        # Remove consecutively repeating tokens and blank tokens
        generated_summary = self.ctc_post_processing(naive_generated_summary)
        decoded_summary_length = sum([self.word_length_tensor[i].item() + 1 for i in generated_summary]) - 1
        #assert decoded_summary_length <= self.sample_desired_length - 1, "Generated summary has a wrong length"
        return generated_summary

    def decode(self, output_logits, source_length):
        # TODO: change the architecture such that source_length is the number of chars in the source
        if output_logits.dtype != torch.float16:
            # CPU is not compatible with float16 on some operation (e.g., softmax)
            output_logits = output_logits.cpu()  # Move everything to CPU for fair comparison (cpu decoding)
        self.dtype = output_logits.dtype
        self.device = output_logits.device
        decoded_summary_list = []
        for i in range(0, len(output_logits)):
            decoded_summary = self.ctc_scope_search_char_length_control(output_logits[i], source_length[i])
            decoded_summary_list.append(decoded_summary)
        return decoded_summary_list

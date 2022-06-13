import torch
from decoding_algorithms.ctc_decoder_base import CTCDecoderBase
import copy


class CTCTestDecoder(CTCDecoderBase):

    def __init__(self, dictionary, decoder_parameters):
        super().__init__(dictionary, decoder_parameters)
        # Sample temporary variable
        self.sample_desired_length = None
        self.id_to_index_dict_list = None
        self.index_to_id_dict_list = None
        self.ctc_sequence_length = None
        self.prob_sequence = None
        self.prev_max_row_index = None
        self.scope_lprob_table = None
        self.transition_tracker_table = None
        self.single_prefix_index_table = None
        # Decoder configuration parameters
        self.force_length = decoder_parameters["force_length"]
        self.use_length_ratio = decoder_parameters["use_length_ratio"]
        self.k = decoder_parameters["k"]  # dummy variable
        self.beam_size = decoder_parameters["beam_size"]
        self.scope = decoder_parameters["scope"]
        self.margin_criteria = decoder_parameters["marg_criteria"]
        self.blank_index = dictionary.blank()
        self.replacing_value = float("-inf")
        self.device = None
        self.dtype = None
        # Assertions on decoder parameters
        assert self.scope > 1, "The scope must be positive integer"
        assert self.beam_size > 0, "Beam size are required to be positive"
        assert self.desired_length > 0, "The desired length should be greater than 0"
        # assert self.beam_size % 2 == 0, "The beam size must be even number"
        # Initialize reusable variables
        self.special_element_tuple_list = [list(range(self.k)), [0] * self.k]
        self.path_num = 2 ** (self.scope - 1)

    def margin_over_prob_table(self, column, dp_log_prob_table, path_num=-1, end_row=0):
        """
        This function marginalize over the first dimension of a k*k*...*k probability tensor based on the given criteria.
        The function returns the chosen token at the first dimension and a modified probability table such that all
        remaining dimensions except from the first dimension is shifted (e.g., the second dimension now becomes the first
        dimension). Furthermore, this probability table copies the the probability except the first axis and repeat it
        across the last dimension of the table (e.g., table[0, 0, 0] == table [0, 0, 1].... if the score is 3.

        prob_table_slot (torch.tensor of size [k]*scope):  the (log) probability of (top k) tokens at last (scope) time steps
        scope (int): dimensions of the prob distribution
        k (int): the size of the word distribution that we are tracking
        criteria (string): in which criteria do we preform the marginalization (currently we only support mean and max)

        returns:
        chosen_token_index (int): the chosen token at the first axis (which is the axis being marginalized)
        new_prob_table (torch.tensor of size [k]*scope): the new probability table after marginalization
        """
        # We take a mean of the probability of all sub-sequences starting from each token in the axis that
        # we are going to marginalize out
        remaining_axis = tuple(range(3, self.scope + 2))  # A tuple of the remaining axis after marginalization
        if self.margin_criteria == "mean":
            # If we are using mean as the marginalization selection criteria
            # we find the sum probability over all sequences starting from each token in the first axis
            # (since each sub-sequence has the same number of elements, we skip the "mean" process to reduce
            # the chance of suffering from underflow.
            # Notice taking exp could lead to underflow of probability.
            sum_old_prob_along_remaining_axis = torch.logsumexp(dp_log_prob_table[0:end_row + 1, column - 1],
                                                                dim=remaining_axis)
            # We find the word with the max-sum probability
            most_probable_word_index = torch.argmax(sum_old_prob_along_remaining_axis, dim=2)
            corresponding_prob = torch.amax(sum_old_prob_along_remaining_axis, dim=2)
        elif self.margin_criteria == "filtered_mean":
            sum_old_prob_along_remaining_axis = torch.logsumexp(dp_log_prob_table[0:end_row + 1, column - 1],
                                                                dim=remaining_axis)
            # We find the word with the max-sum probability
            non_inf_sum = (dp_log_prob_table[0:end_row + 1, column - 1] != float("-inf")).long().sum(remaining_axis)
            sum_old_prob_along_remaining_axis = sum_old_prob_along_remaining_axis - non_inf_sum.log()
            sum_old_prob_along_remaining_axis = torch.nan_to_num(sum_old_prob_along_remaining_axis, float("-inf"))
            most_probable_word_index = torch.argmax(sum_old_prob_along_remaining_axis, dim=2)
            corresponding_prob = torch.amax(sum_old_prob_along_remaining_axis, dim=2)
        elif self.margin_criteria == "max":
            # If we are using max as the select criteria, we select the token in the first axis that can lead to the
            # sub-sequence with the maximum probability
            max_old_prob_along_remaining_axis = torch.amax(dp_log_prob_table[0:end_row + 1, column - 1],
                                                           dim=remaining_axis)
            most_probable_word_index = torch.argmax(max_old_prob_along_remaining_axis, dim=2)
            corresponding_prob = torch.amax(max_old_prob_along_remaining_axis, dim=2)
        elif self.margin_criteria == "beam_search":
            max_old_prob_along_remaining_axis = dp_log_prob_table[0:end_row + 1, column - 1]
            most_probable_word_index = torch.argmax(max_old_prob_along_remaining_axis, dim=2)
            corresponding_prob = torch.amax(max_old_prob_along_remaining_axis, dim=2)
        else:
            most_probable_word_index = None
            raise NotImplementedError("Haven't designed other evaluation criteria")
        # We make a clone of the old probability such that we can modify this tensor
        # Notice we only select the sub-tensor starting from the selected token
        prob_size = dp_log_prob_table.size()
        new_prob_index = [torch.arange(0, end_row + 1).repeat(prob_size[2], 1).transpose(0, 1).flatten(),
                          column - 1,
                          torch.arange(0, prob_size[2]).repeat(end_row + 1, 1).flatten(),
                          most_probable_word_index.view(-1)]
        new_probability = dp_log_prob_table[new_prob_index].clone()
        new_probability = new_probability.reshape([end_row + 1, path_num] + list(prob_size[4:]))
        new_probability = new_probability.unsqueeze(-1)
        # the new dimension is in the wrong shape, we repeat it for k times to keep our data structure
        # we calculate the desired repeat-index to repeat for k times along the last dimension
        repeat_index = tuple([1, 1] + (self.scope - 1) * [1] + [self.k])
        # Now we have a probability distribution that has the same dimension as what we have
        # at the beginning and a polished last dimension where we repeat the probability for subsequences
        # ending at the second last dimensions.
        new_prob_table = new_probability.repeat(repeat_index)
        return most_probable_word_index, new_prob_table, corresponding_prob

    def top_k_filtering(self, logits, prev_index_to_id_dict, prev_id_to_index_dict, blank_token_id=2):
        """
        Get the top-k most probable token and their corresponding probabilities
        logits (tensor): the logits returned by the model at the current time step
        k (int): the number of the top tokens that we desire
        blank_token_id (int): the id of the blank token

        Return:
            values (tensor of size k): the probability of the most probable tokens, with the first one set to be the blank token
            index_to_id_dict (dict of size k): a dictionary mapping from the element index of the values vector to their real id
            repeated_element_index_list (list): a list of the index (row, column) indicating the repeating index
        """

        values, ids = logits.topk(self.k)
        # print("pure top k values")
        # print(values)
        if blank_token_id in ids:
            # If the blank token is one of the top k token
            temp_index = ((ids == blank_token_id).nonzero(as_tuple=True)[0]).item()  # the index of the blank token
            temp_value = values[temp_index].clone()  # the prob_value of the blank token
            values[temp_index] = values[0]  # swap the index between the top token and the blank token
            ids[temp_index] = ids[0]
            values[0] = temp_value  # we place the blank token's probability as the first element of the value vector
            ids[0] = blank_token_id
            # print("blank in ids")
            # print(values)
        else:
            # If the blank token is not one of the top k tokens
            values[1:self.k] = values[0:self.k - 1].clone()  # we perform a shift and drop the last top token
            ids[1:self.k] = ids[0:self.k - 1].clone()
            values[0] = logits[blank_token_id]
            ids[0] = blank_token_id
            # print("blank not in ids")
            # print(values)
        index_to_id_dict = {}
        id_to_index_dict = {}
        repeated_element_index_list = []  # index of repeated elements
        non_repeated_element_non_special_index_list = []  # index of non-repeated elements

        for index in range(0, self.k):
            # We create a dictionary mapping from the index of each value in the values tensor to it's corresponding id
            # in the logits tensor
            current_dict_id = ids[index].item()
            index_to_id_dict[index] = current_dict_id
            id_to_index_dict[current_dict_id] = index
            if prev_index_to_id_dict is not None:
                # If we are not operating on the first dictionary
                if current_dict_id in prev_index_to_id_dict.values():
                    prev_dict_element_index = prev_id_to_index_dict[current_dict_id]
                    repeated_element_index_list.append((prev_dict_element_index, index))
        repeated_or_special_element_index_list = copy.deepcopy(repeated_element_index_list)

        for i in range(0, self.k):
            for j in range(0, self.k):
                if (i, j) not in repeated_element_index_list:
                    if index_to_id_dict[j] != blank_token_id:
                        non_repeated_element_non_special_index_list.append((i, j))
                    else:
                        repeated_or_special_element_index_list.append((i, j))
        # Notice if the repeated token is not in the top-k token dictionary of the current time step, we don't include
        # it in the remove_repeated_element_mask.
        assert len(repeated_or_special_element_index_list) == len(
            set(repeated_or_special_element_index_list)), "there are repeated index!"
        assert len(non_repeated_element_non_special_index_list) == len(
            set(non_repeated_element_non_special_index_list)), "there are repeated index!"
        assert -1 not in index_to_id_dict.keys()
        assert -1 not in index_to_id_dict.values()
        repeated_or_special_element_index_list = [tuple([x[0] for x in repeated_or_special_element_index_list]),
                                                  tuple([x[1] for x in repeated_or_special_element_index_list])]
        non_repeated_element_non_special_index_list = [
            tuple([x[0] for x in non_repeated_element_non_special_index_list]),
            tuple([x[1] for x in non_repeated_element_non_special_index_list])]
        return values, index_to_id_dict, id_to_index_dict, repeated_or_special_element_index_list, non_repeated_element_non_special_index_list

    def ctc_length_control_initialization(self, logits, device):
        """
        Perform the initialization of the dynamic programming ctc length control algorithm
        Input:
            logits: Tensor(sequence_length*num_words) logits over different words (we force batch size to be 1 for now)
            desired_length: (int) the desired length of the summary.
            scope: (int) the length of the (ctc token) subsequence probability distribution that we are tracking
            k: (int) the number of most probable words that we keep in each position
            device: the device to store probabilities & perform calculation
        Return:
            prob_sequence: (Tensor) The probability over all possible words at each time step
            ctc_sequence_length: (int) The length of the ctc sequence
            dp_token_table: (Tensor) The determined token at each table slot
            dp_dict_table: (Tensor) The dictionary mapping from word index to word id at each table slot
            dp_prob_table: (Tensor) The table to store the probability of generating summaries of various lengths
                            given different length of the logit sequence.
            dp_marg_prob_table: (Tensor) The table to store the marginalized version of dp_prob_table
        """
        prob_sequence = torch.nn.functional.softmax(logits, dim=-1)  # Get the log probability from logits
        # Notice scope = 1 means we only consider the probability distribution over words for the current time step
        # Notice k = 10 means we only care about the top 10 words at each time step
        ctc_sequence_length = len(prob_sequence)  # The length of the ctc output sequence.
        if self.scope > ctc_sequence_length:
            # If the scope exceeds ctc_sequence_length
            raise ValueError("The scope to reduce cannot exceed the length of the ctc output sequence")
        elif self.scope < 1:
            raise ValueError("The scope must be positive integer")
        # token table dimension store the determined token at each table slot
        # dict table dimension store the dictionary mapping from index (e.g., 1~k) to token id (e.g., 1~50000)
        # prob_dimensions determines how many & how large distributions do we trace
        dp_prob_dimensions = [self.desired_length, ctc_sequence_length, self.path_num] + self.scope * [self.k]
        dp_marg_prob_dimensions = [self.desired_length, ctc_sequence_length, self.path_num] + self.scope * [self.k]
        dp_prefix_id_table = torch.zeros([self.desired_length, ctc_sequence_length, self.path_num, ctc_sequence_length],
                                         dtype=torch.long, device=device) - 1
        dp_index_to_id_dict_list = [
                                       -1] * ctc_sequence_length  # The main table to store the dictionary mapping from token index to token id
        dp_id_to_index_dict_list = [-1] * ctc_sequence_length
        dp_prob_table = torch.zeros(dp_prob_dimensions, dtype=prob_sequence.dtype,
                                    device=device)  # The main table to store DP (dynamic programming) probability result
        # We perform a minus to indicate that this table cannot be used without initialzation
        # Error would occour if the table is used without initialization.
        dp_marg_prob_table = torch.zeros(dp_marg_prob_dimensions, dtype=prob_sequence.dtype,
                                         device=device) - 1  # The main table to store DP (dynamic programming) probability result
        dp_log_token_corresponding_prob_table = torch.zeros([self.desired_length, ctc_sequence_length, self.path_num],
                                                            dtype=prob_sequence.dtype,
                                                            device=device)
        dp_log_prefix_prob_table = torch.zeros([self.desired_length, ctc_sequence_length, self.path_num],
                                               dtype=prob_sequence.dtype,
                                               device=device)

        return prob_sequence, ctc_sequence_length, dp_prefix_id_table, dp_index_to_id_dict_list, \
               dp_id_to_index_dict_list, dp_prob_table, dp_marg_prob_table, dp_log_token_corresponding_prob_table, \
               dp_log_prefix_prob_table

    def get_special_tokens_prob(self, special_token_ids, current_index_to_id_dict, current_id_to_index_dict,
                                current_filtered_log_prob, replacing_value=float("-inf")):
        only_special_cloned_prob = current_filtered_log_prob.clone()
        for i in current_index_to_id_dict.values():
            if i not in special_token_ids:
                token_index = current_id_to_index_dict[i]
                # add the index of the special token to the list
                only_special_cloned_prob[token_index] = replacing_value
        return only_special_cloned_prob

    def get_non_special_tokens_prob(self, special_token_ids, current_index_to_id_dict, current_id_to_index_dict,
                                    current_filtered_log_prob, replacing_value=float("-inf")):
        only_non_special_cloned_prob = current_filtered_log_prob.clone()
        for i in current_index_to_id_dict.values():
            if i in special_token_ids:
                token_index = current_id_to_index_dict[i]
                # add the index of the special token to the list
                only_non_special_cloned_prob[token_index] = replacing_value
        return only_non_special_cloned_prob

    def row_inference(self, column, prev_max_row_index, reshaping_index, remaining_scope, split_point,
                      only_special_cloned_prob, only_non_special_cloned_prob,
                      non_special_non_repeated_transition_matrix, special_or_repeated_transition_matrix,
                      dp_log_prob_table,
                      dp_log_prefix_prob_table, dp_prefix_id_table, prev_prob_table, chosen_token_index_list,
                      chosen_token_id_list, test_table=None):
        """
        Perform actual table filling for each rows.
        """
        if column == 0:
            # For the first row, we initialize all non-special tokens with -inf probability
            # we only initialize the value to the first rwo to avoid repeated path
            dp_log_prob_table[0, column, 0] += only_special_cloned_prob[reshaping_index]
            dp_log_prob_table[0, column, 1:] += float("-inf")
            test_table[0, column, 0, 0] = 0
            # For the second row, we initialize it with the probability of non-special tokens
            # we only initialize the value to the first row to avoid repeated path
            dp_log_prob_table[1, column, 0] += only_non_special_cloned_prob[reshaping_index]
            dp_log_prob_table[1, column, 1:] += float("-inf")
            test_table[1, column, 0, 0] = 1
        else:
            # For other columns, we first solve the first row and the last row since they do not require case split.
            dp_log_prob_table[0, column, 0] = prev_prob_table[0, column - 1, 0] + only_special_cloned_prob[
                reshaping_index]
            dp_log_prob_table[0, column, 1:] = float("-inf")
            test_table[0, column, 0] = test_table[0, column - 1, 0]
            test_table[0, column, 0, column] = 0
            if column + 1 < self.desired_length:
                # If we still have space for pure expansion
                dp_log_prob_table[column + 1, column, 0] = prev_prob_table[column, column - 1, 0] \
                                                           + non_special_non_repeated_transition_matrix[reshaping_index]
                dp_log_prob_table[column + 1, column, 1:] = float("-inf")
                test_table[column + 1, column, 0] = test_table[column, column - 1, 0]
                test_table[column + 1, column, 0, column] = column + 1
            # For other rows (i.e., middle rows)
            # We repeat the assignment for a couple times to fill up the table slot
            if remaining_scope > 0:
                # If we still have enough position to store the paths and probabilities
                # We first determine the first half probability, which goes from diagonal-neighbouring slot.
                dp_log_prob_table[1:prev_max_row_index + 1, column, 0:2 * split_point] = torch.cat(
                    [prev_prob_table[:prev_max_row_index, column - 1, 0:split_point] +
                     non_special_non_repeated_transition_matrix[reshaping_index],
                     prev_prob_table[1:prev_max_row_index + 1, column - 1, 0:split_point] +
                     special_or_repeated_transition_matrix[reshaping_index]],
                    dim=1)
                dp_log_prob_table[1:prev_max_row_index + 1, column, 2 * split_point:] = float("-inf")
                test_table[1:prev_max_row_index + 1, column, 0:split_point] = test_table[:prev_max_row_index,
                                                                              column - 1, 0:split_point]
                test_table[1:prev_max_row_index + 1, column, split_point:2 * split_point] = test_table[
                                                                                            1:prev_max_row_index + 1,
                                                                                            column - 1, 0:split_point]
                test_table[1:prev_max_row_index + 1, column, 0:2 * split_point, column] += torch.arange(1,
                                                                                                        prev_max_row_index + 1).unsqueeze(
                    1) + 1
            else:
                # If we are running out of space and marginalization was performed
                # We first store the best paths in previous table slots
                prev_best_path_prob, prev_best_path_index = dp_log_prefix_prob_table[:prev_max_row_index + 1,
                                                            column - 1].topk(split_point, dim=1)
                # upper path probability records the probability transiting from the diagonal-neighbouring slots.
                diagonal_neighbouring_row_index = torch.arange(0, prev_max_row_index).repeat(split_point, 1).transpose(
                    0, 1).flatten()
                row_neighbouring_row_index = torch.arange(1, prev_max_row_index + 1).repeat(split_point, 1).transpose(0,
                                                                                                                      1).flatten()
                diagonal_neighbouring_prob = prev_prob_table[diagonal_neighbouring_row_index, column - 1,
                                                             prev_best_path_index[0:prev_max_row_index, :].view(
                                                                 -1)].view(
                    [prev_max_row_index, split_point] + self.scope * [self.k])
                # lower path probability records the probability transiting from the row-neighbouring slots.
                row_neighbouring_prob = prev_prob_table[row_neighbouring_row_index, column - 1,
                                                        prev_best_path_index[1:prev_max_row_index + 1, :].view(
                                                            -1)].view([prev_max_row_index, split_point] + self.scope * [self.k])

                dp_log_prob_table[1:prev_max_row_index + 1, column] = \
                    torch.cat([diagonal_neighbouring_prob + non_special_non_repeated_transition_matrix[reshaping_index],
                               row_neighbouring_prob + special_or_repeated_transition_matrix[reshaping_index]], dim=1)

                dp_prefix_id_table[0, column] = dp_prefix_id_table[0, column - 1]
                dp_prefix_id_table[0, column, :, column] = chosen_token_id_list[0]
                if column + 1 < self.desired_length:
                    # If we still have pure expansion
                    dp_prefix_id_table[column + 1, column] = dp_prefix_id_table[column, column - 1]
                    dp_prefix_id_table[column + 1, column, :, column] = chosen_token_id_list[column]
                    # If it's at middle rows, it cannot perfectly inherits and therefore needs filtering

                dp_prefix_id_table[1:prev_max_row_index + 1, column, 0:split_point] = \
                    dp_prefix_id_table[
                        diagonal_neighbouring_row_index, column - 1, prev_best_path_index[0:prev_max_row_index].view(
                            -1)].view(prev_max_row_index, split_point, -1)
                dp_prefix_id_table[1:prev_max_row_index + 1, column, 0:split_point, column] = \
                    chosen_token_id_list[
                        diagonal_neighbouring_row_index, prev_best_path_index[0:prev_max_row_index].view(-1)].view(
                        prev_max_row_index, split_point)

                dp_prefix_id_table[1:prev_max_row_index + 1, column, split_point:] = \
                    dp_prefix_id_table[
                        row_neighbouring_row_index, column - 1, prev_best_path_index[1:prev_max_row_index + 1].view(
                            -1)].view(prev_max_row_index, split_point, -1)
                dp_prefix_id_table[1:prev_max_row_index + 1, column, split_point:, column] = \
                    chosen_token_id_list[
                        row_neighbouring_row_index, prev_best_path_index[1:prev_max_row_index + 1].view(-1)].view(
                        prev_max_row_index, split_point)

    def column_inference(self, column, current_log_prob, prev_index_to_id_dict, prev_id_to_index_dict,
                         ctc_blank_token_id, dp_index_to_id_dict_list, dp_id_to_index_dict_list,
                         special_token_ids, replacing_value, dp_log_prob_table,
                         dp_log_marg_prob_table, dp_log_prefix_prob_table, dp_prefix_id_table, test_table=None):
        """
        Perform table (prob table and prefix table) filling for a single column
        """
        remaining_scope = self.scope - column  # The remaining unoccupied dimension in the table slots of the previous column
        # The maximum index of the generated summary at the previous time step
        # For example, at the 4-th column, it can generate at most length-5 summary, so it's previous time step can
        # generate at most length-4 = length(column) summary, which has a corresponding index of 4
        prev_max_row_index = min(column, self.desired_length - 1)
        if remaining_scope > 1:
            split_point = 2 ** (column - 1)
        else:
            split_point = 2 ** (self.scope - 2)
        # Get the filtered top-probabilities and dictionary mapping between token index and token id
        # Get the probability of the top k tokens at the current time step, also the mapping mask between the previous
        # time step and the current time step (top k tokens could be different at different time step).
        current_filtered_log_prob, current_index_to_id_dict, current_id_to_index_dict, \
        repeated_or_special_element_index_list, non_repeated_element_non_special_index_list = \
            self.top_k_filtering(current_log_prob, prev_index_to_id_dict, prev_id_to_index_dict,
                                 blank_token_id=ctc_blank_token_id)
        dp_index_to_id_dict_list[column] = current_index_to_id_dict  # Store the dictionary to list
        dp_id_to_index_dict_list[column] = current_id_to_index_dict
        store_token_column = column - self.scope  # The column to store the determined token during marginalization
        only_special_cloned_prob = self.get_special_tokens_prob(
            special_token_ids, current_index_to_id_dict, current_id_to_index_dict, current_filtered_log_prob,
            replacing_value=replacing_value)
        only_non_special_cloned_prob = self.get_non_special_tokens_prob(
            special_token_ids, current_index_to_id_dict, current_id_to_index_dict, current_filtered_log_prob,
            replacing_value=replacing_value)
        # Filter the list of repeated element indexs such that the remaining index tuples in the list doesn't not
        # have special token index as the second element
        non_special_non_repeated_transition_matrix = current_filtered_log_prob.expand(self.k, self.k).clone()
        non_special_non_repeated_transition_matrix[repeated_or_special_element_index_list] = replacing_value
        special_or_repeated_transition_matrix = current_filtered_log_prob.expand(self.k, self.k).clone()
        special_or_repeated_transition_matrix[non_repeated_element_non_special_index_list] = replacing_value
        chosen_token_index_list = torch.zeros([prev_max_row_index + 1, self.path_num], dtype=torch.long,
                                              device=current_log_prob.device)
        chosen_token_id_list = torch.zeros([prev_max_row_index + 1, self.path_num], dtype=torch.long,
                                           device=current_log_prob.device)
        if remaining_scope > 0:
            # If we have enough scope to store the current probability without requiring marginalization
            reshaping_index = tuple((...,) + (None,) * (remaining_scope - 1))
            prev_prob_table = dp_log_prob_table
        else:
            # If we don't have enough scope to store the current probability, we will perform marginalization on the
            # previous prob distributions such that the remaining scope becomes 1.
            chosen_token_index_list, new_prob_table, prefix_prob = \
                self.margin_over_prob_table(column, dp_log_prob_table,
                                            path_num=self.path_num, end_row=prev_max_row_index)
            reshaping_index = tuple((...,) + (None,) * (1 - 1))
            prev_dict = dp_index_to_id_dict_list[store_token_column]
            for i in range(0, prev_max_row_index + 1):
                for j in range(0, self.path_num):
                    chosen_token_id_list[i, j] = (prev_dict[chosen_token_index_list[i, j].item()])
            # Add the marginalized probability to prob table
            dp_log_marg_prob_table[:prev_max_row_index + 1, column - 1] = new_prob_table
            # Add the token probability to token prob table
            dp_log_prefix_prob_table[:prev_max_row_index + 1, column - 1] = prefix_prob
            prev_prob_table = dp_log_marg_prob_table
        self.row_inference(column, prev_max_row_index, reshaping_index, remaining_scope, split_point,
                           only_special_cloned_prob, only_non_special_cloned_prob,
                           non_special_non_repeated_transition_matrix, special_or_repeated_transition_matrix,
                           dp_log_prob_table,
                           dp_log_prefix_prob_table, dp_prefix_id_table, prev_prob_table, chosen_token_index_list,
                           chosen_token_id_list, test_table=test_table)

    def my_remove_adjacent(self, nums):
        return [a for a, b in zip(nums, nums[1:] + [not nums[-1]]) if a != b]

    def ctc_length_control(self, logits, device=torch.device('cuda'), ctc_blank_token_id=4):
        """
        This function perform length control on the output CTC logits of the model decoder.

        Inputs:
            logits: Tensor(sequence_length*num_words) logits over different words (we force batch size to be 1 for now)
            length_control: (str) the method that we are going to adopt to perform length control.
            desired_length: (int) the desired length of the summary.
            scope: (int) the length of the (ctc token) subsequence probability distribution that we are tracking
            k: (int) the number of most probable words that we keep in each position
            marg_criteria: (string) in which criteria do we perform marginalization on old logits.
            ctc_blank_token_id: (int) the id of the ctc blank token.
            ctc_eos_token_id: (int) the id of the ctc eos token
            device: the device to store probabilities & perform calculation

        There are five different cases:
            (Notice first row represents a resulting summary of length 0, first column represents given first logits)
            1) If the current table slot is the 1-th row & 1-th column, the only possible token is [blank] token. We
                append the [blank] token to the token table to indicate that the ctc token for this table slot has been
                determined.
                Notice we set all elements in current_table_slot[0] to be the probability of this determined token and
                other elements to be 0 to indicate that the only non-zero probability happens at the [blank] token.
            2) If the current table slot is the 1-th row & i-th column (i!=1), similarly to the previous scenario,
                the only posible token is [blank] token
            3) If the current table slot is the 2-th row & 1-th column, similarly to previous scenario, we don't have
                previous prob, so we select the top k non-blank probability of from the current probability and append
                them to the corresponding elements of the current table_slot. (e.g., table_slot[1] = current_prob[1].
            4) If the current table slot is the (i+1)-th row & i-th column, we must continue from the i-th row and the
                (i-1)-th column and generate a new token which is not repeated as the token in the i-th row and the (i-1)
                -th column.
            5) If the current table slot is i-th row & j-th column, where i < j, there are two choices to pick.
                The first one is to
                continue with the subsequence ended at table slot (i-1, j-1) and generate a non-repeated token. The second
                choice is to continue with the subsequence ended at table slot (i, j-1) and generate either a blank token
                or a repeated token.
        """
        ############################################# Initialization #############################################
        self.desired_length += 1
        # desired_length = round(0.34*len(logits))+1
        prob_sequence, ctc_sequence_length, dp_prefix_id_table, dp_index_to_id_dict_list, dp_id_to_index_dict_list, \
        dp_log_prob_table, dp_log_marg_prob_table, dp_log_token_corresponding_prob_table, dp_log_prefix_prob_table \
            = self.ctc_length_control_initialization(logits, device)
        _, naive_summary = prob_sequence.max(-1)
        naive_summary = self.my_remove_adjacent(naive_summary.tolist())
        naive_summary = list(filter((ctc_blank_token_id).__ne__, naive_summary))
        use_shorter_summary = (len(naive_summary) <= self.desired_length - 1) and (not self.force_length)
        source_too_short = (
                    len(prob_sequence) <= self.desired_length - 1)  # If the source sentence/CTC output sequence does not have enough length
        if use_shorter_summary or source_too_short:
            # The untruncated summary already has a satisfied length
            self.desired_length -= 1
            return naive_summary
        special_token_ids = [
            ctc_blank_token_id]  # A list of token ids that does not contribute to the real words generation
        test_table = torch.zeros([self.desired_length, ctc_sequence_length, self.path_num, ctc_sequence_length],
                                 dtype=torch.long) - 1
        replacing_value = float("-inf")
        ############################################# Main Loop ##################################################
        for column in range(0, ctc_sequence_length):
            # For each column, we calculate the desired probability and prefix for each allowable table slot.
            current_log_prob = prob_sequence[column].log()
            if column == 0:
                prev_index_to_id_dict = None
                prev_id_to_index_dict = None
            else:
                prev_index_to_id_dict = dp_index_to_id_dict_list[column - 1]
                prev_id_to_index_dict = dp_id_to_index_dict_list[column - 1]
            self.column_inference(column, current_log_prob, prev_index_to_id_dict, prev_id_to_index_dict,
                                   ctc_blank_token_id, dp_index_to_id_dict_list,
                                  dp_id_to_index_dict_list,
                                  special_token_ids, replacing_value, dp_log_prob_table,
                                  dp_log_marg_prob_table, dp_log_prefix_prob_table, dp_prefix_id_table,
                                  test_table=test_table)

        # Now we have a partially determined ctc token, we still need to marginalize the remaining probabilities in the
        # dp probability table to fill in the whole token table.
        # The first token can be obtained from the next function so we only include the last few tokens.
        prob_table_slot = dp_log_prob_table[self.desired_length - 1, ctc_sequence_length - 1]
        maximum_index = (prob_table_slot == torch.amax(prob_table_slot)).nonzero()[0]
        postfix_sentence = maximum_index[1:]
        generated_sentence = dp_prefix_id_table[self.desired_length - 1, ctc_sequence_length - 1, maximum_index[0]].tolist()
        for i in range(0, len(postfix_sentence)):
            store_token_column = ctc_sequence_length - self.scope + i
            current_index = postfix_sentence[i].item()
            current_dict = dp_index_to_id_dict_list[store_token_column]
            current_id = current_dict[current_index]
            generated_sentence += [current_id]
        generated_sentence = self.my_remove_adjacent(generated_sentence)
        generated_sentence = list(filter((ctc_blank_token_id).__ne__, generated_sentence))
        generated_sentence = list(filter((-1).__ne__, generated_sentence))
        assert len(generated_sentence) == self.desired_length - 1, "Generated summary has a wrong length"
        self.desired_length -= 1
        return generated_sentence

    def decode(self, output_logits, source_length):
        """
        Decoding function for the CTC scope search decoder.
        """
        if output_logits.dtype != torch.float16:
            output_logits = output_logits.cpu()  # Move everything to CPU for fair comparison (cpu decoding)
        self.dtype = output_logits.dtype
        self.device = output_logits.device
        decoded_summary_list = []
        for i in range(0, len(output_logits)):
            decoded_summary = self.ctc_length_control(output_logits[i], device=torch.device('cpu'))
            decoded_summary_list.append(decoded_summary)
        return decoded_summary_list
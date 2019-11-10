# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A powerful dynamic attention wrapper object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops, math_ops, linalg_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops, variable_scope
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _compute_attention, LuongAttention, \
    _BaseAttentionMechanism, _luong_score, _bahdanau_score, AttentionMechanism, _maybe_mask_score, _prepare_memory
from tensorflow.python.layers.core import Dense

import tensorflow as tf

__all__ = [
    "MyAttentionWrapper",
    "AttentionWrapperState",
]

_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access


# edit by acharkq
class AttentionWrapperState(
    collections.namedtuple("AttentionWrapperState",
                           ("cell_state", "attention", "time", "alignments",
                            "alignment_history", "attention_state", "last_choice"))):

    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, ops.Tensor) and isinstance(new, ops.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
            with_same_shape,
            self,
            super(AttentionWrapperState, self)._replace(**kwargs))


class AttentionWrapperState_v2(
    collections.namedtuple("AttentionWrapperState_v2",
                           ("cell_state", "attention", "time", "alignments",
                            "alignment_history", "last_choice"))):

    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, ops.Tensor) and isinstance(new, ops.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
            with_same_shape,
            self,
            super(AttentionWrapperState_v2, self)._replace(**kwargs))


def masked_mean_pooling(input, sequence_masks, max_sequence_length):
    '''do masked mean pooling on the sequence_length dimension of a tensor with shape=[batch_size, sequence_length]'''
    sequence_mask = tf.sequence_mask(sequence_masks, maxlen=max_sequence_length, dtype=tf.float32)
    tmp = tf.multiply(input, sequence_mask) / tf.reduce_sum(sequence_mask, axis=1, keepdims=True)
    pooled = tf.reduce_sum(tmp, axis=1)
    return pooled


class MyAttentionWrapper(AttentionWrapper):
    """Wraps another `RNNCell` with attention.
    """

    # edit by acharkq
    def __init__(self,
                 cell,
                 attention_mechanism,
                 sot,
                 attention=True,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None):

        self.sot = sot
        self._attention = attention
        super(MyAttentionWrapper, self).__init__(cell=cell, attention_mechanism=attention_mechanism,
                                                 attention_layer_size=attention_layer_size,
                                                 alignment_history=alignment_history, cell_input_fn=cell_input_fn,
                                                 output_attention=output_attention,
                                                 initial_cell_state=initial_cell_state, name=name)

    # edit by acharkq
    @property
    def state_size(self):
        """The `state_size` property of `AttentionWrapper`.

        Returns:
          An `AttentionWrapperState` tuple containing shapes used by this object.
        """
        return AttentionWrapperState(
            cell_state=self._cell.state_size,
            time=tensor_shape.TensorShape([]),
            attention=self._attention_layer_size,
            alignments=self._item_or_tuple(
                a.alignments_size for a in self._attention_mechanisms),
            attention_state=self._item_or_tuple(
                a.state_size for a in self._attention_mechanisms),
            alignment_history=self._item_or_tuple(
                () for _ in self._attention_mechanisms),
            last_choice=tensor_shape.TensorShape([]))  # sometimes a TensorArray

    # edit by acharkq
    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.

        **NOTE** Please see the initializer documentation for details of how
        to call `zero_state` if using an `AttentionWrapper` with a
        `BeamSearchDecoder`.

        Args:
          batch_size: `0D` integer tensor: the batch size.
          dtype: The internal state data type.

        Returns:
          An `AttentionWrapperState` tuple containing zeroed out tensors and,
          possibly, empty `TensorArray` objects.

        Raises:
          ValueError: (or, possibly at runtime, InvalidArgument), if
            `batch_size` does not match the output size of the encoder passed
            to the wrapper object at initialization time.
        """
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                    "When calling zero_state of AttentionWrapper %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and the requested batch size.  Are you using "
                    "the BeamSearchDecoder?  If so, make sure your encoder output has "
                    "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
                    "the batch_size= argument passed to zero_state is "
                    "batch_size * beam_width.")
            with ops.control_dependencies(
                    self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                    lambda s: array_ops.identity(s, name="checked_cell_state"),
                    cell_state)
            return AttentionWrapperState(
                cell_state=cell_state,
                time=array_ops.zeros([], dtype=dtypes.int32),
                attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                              dtype),
                alignments=self._item_or_tuple(
                    attention_mechanism.initial_alignments(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms),
                attention_state=self._item_or_tuple(
                    attention_mechanism.initial_state(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms),
                alignment_history=self._item_or_tuple(
                    tensor_array_ops.TensorArray(dtype=dtype, size=0,
                                                 dynamic_size=True)
                    if self._alignment_history else ()
                    for _ in self._attention_mechanisms),
                last_choice=array_ops.fill([batch_size], self.sot))

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.

        - Step 1: Mix the `inputs` and previous step's `attention` output via
          `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
          and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).

        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time step.
          state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.

        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:

          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `AttentionWrapperState`
             containing the state calculated at this time step.

        Raises:
          TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead." % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        if self._attention:
            cell_inputs = self._cell_input_fn(inputs, state.attention)
        else:
            cell_inputs = inputs

        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
                cell_output.shape[0].value or array_ops.shape(cell_output)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the query (decoder output).  Are you using "
                "the BeamSearchDecoder?  You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with ops.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = array_ops.identity(
                cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history = state.alignment_history
        else:
            previous_attention_state = [state.attention_state]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments, next_attention_state = _compute_attention(
                attention_mechanism, cell_output, previous_attention_state[i],
                self._attention_layers[i] if self._attention_layers else None)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)

        attention = array_ops.concat(all_attentions, 1)
        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            attention_state=self._item_or_tuple(all_attention_states),
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(maybe_all_histories),
            last_choice=state.last_choice)

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state


class MyAttentionWrapper_v2(MyAttentionWrapper):
    """
    this version let the query be the embedding of the previous choice
    """

    # edit by acharkq
    @property
    def state_size(self):
        """The `state_size` property of `AttentionWrapper`.

        Returns:
          An `AttentionWrapperState` tuple containing shapes used by this object.
        """
        return AttentionWrapperState_v2(
            cell_state=self._cell.state_size,
            time=tensor_shape.TensorShape([]),
            attention=self._attention_layer_size,
            alignments=self._item_or_tuple(
                a.alignments_size for a in self._attention_mechanisms),
            alignment_history=self._item_or_tuple(
                () for _ in self._attention_mechanisms),
            last_choice=tensor_shape.TensorShape([]))  # sometimes a TensorArray

    # edit by acharkq
    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.

        **NOTE** Please see the initializer documentation for details of how
        to call `zero_state` if using an `AttentionWrapper` with a
        `BeamSearchDecoder`.

        Args:
          batch_size: `0D` integer tensor: the batch size.
          dtype: The internal state data type.

        Returns:
          An `AttentionWrapperState` tuple containing zeroed out tensors and,
          possibly, empty `TensorArray` objects.

        Raises:
          ValueError: (or, possibly at runtime, InvalidArgument), if
            `batch_size` does not match the output size of the encoder passed
            to the wrapper object at initialization time.
        """
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                    "When calling zero_state of AttentionWrapper %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and the requested batch size.  Are you using "
                    "the BeamSearchDecoder?  If so, make sure your encoder output has "
                    "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
                    "the batch_size= argument passed to zero_state is "
                    "batch_size * beam_width.")
            with ops.control_dependencies(
                    self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                    lambda s: array_ops.identity(s, name="checked_cell_state"),
                    cell_state)
            return AttentionWrapperState_v2(
                cell_state=cell_state,
                time=array_ops.zeros([], dtype=dtypes.int32),
                attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                              dtype),
                alignments=self._item_or_tuple(
                    attention_mechanism.initial_alignments(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms),
                alignment_history=self._item_or_tuple(
                    tensor_array_ops.TensorArray(dtype=dtype, size=0,
                                                 dynamic_size=True)
                    if self._alignment_history else ()
                    for _ in self._attention_mechanisms),
                last_choice=array_ops.fill([batch_size], self.sot))

    def call(self, inputs, state):
        if not isinstance(state, AttentionWrapperState_v2):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead." % type(state))
        if self._is_multi:
            previous_alignment_history = state.alignment_history
        else:
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        maybe_all_histories = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments, next_attention_state = _compute_attention(
                attention_mechanism, inputs, attention_state=None,
                attention_layer=self._attention_layers[i] if self._attention_layers else None)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)

        attention = array_ops.concat(all_attentions, 1)

        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
                cell_output.shape[0].value or array_ops.shape(cell_output)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the query (decoder output).  Are you using "
                "the BeamSearchDecoder?  You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with ops.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = array_ops.identity(
                cell_output, name="checked_cell_output")

        next_state = AttentionWrapperState_v2(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(maybe_all_histories),
            last_choice=state.last_choice)

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state


class MyBahdanauAttention(_BaseAttentionMechanism):

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 name="MyBahdanauAttention"):
        if probability_fn is None:
            probability_fn = nn_ops.softmax
        if dtype is None:
            dtype = dtypes.float32
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(MyBahdanauAttention, self).__init__(
            query_layer=Dense(
                num_units, name="query_layer", use_bias=False, dtype=dtype, _reuse=tf.AUTO_REUSE, _scope='query_scope'),
            memory_layer=Dense(
                num_units, name="memory_layer", use_bias=False, dtype=dtype, _reuse=tf.AUTO_REUSE,
                _scope='memory_scope'),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._memory_sequence_length = memory_sequence_length
        self._num_units = num_units
        self._normalize = normalize
        self._name = name

    def __call__(self, query, state, softmaxed=True):
        """Score the query based on the keys and values.

        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _bahdanau_score(processed_query, self._keys, self._normalize)
        if softmaxed:
            alignments = self._probability_fn(score, state)
            next_state = alignments
            return alignments, next_state
        score = masked_mean_pooling(score, self._memory_sequence_length, int(self._values.shape[1]))
        return score

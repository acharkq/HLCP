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
"""A decoder that performs beam search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.seq2seq import BeamSearchDecoder, BeamSearchDecoderOutput
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import _mask_probs, _get_scores, _tensor_gather_helper, \
    _maybe_tensor_gather_helper, _beam_search_step
import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, gen_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
import tensorflow as tf

# edit by acharkq
# last_choice: the decision of the decoder on the last time_step
class BeamSearchDecoderState(
    collections.namedtuple("BeamSearchDecoderState",
                           ("cell_state", "log_probs", "finished", "lengths", "last_choice"))):
    pass



class MyBeamSearchDecoder(BeamSearchDecoder):

    def __init__(self,
                 cell,
                 embedding,
                 start_tokens,
                 end_token,
                 initial_state,
                 beam_width,
                 lookup_table,
                 output_layer=None,
                 hie=True,
                 length_penalty_weight=0.0):
        # edit by acharkq
        self.lookup_table = lookup_table
        self.hie = hie
        super(MyBeamSearchDecoder, self).__init__(cell,
                                                  embedding,
                                                  start_tokens,
                                                  end_token,
                                                  initial_state,
                                                  beam_width,
                                                  output_layer=output_layer,
                                                  length_penalty_weight=length_penalty_weight)

    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.

        Args:
          time: scalar `int32` tensor.
          inputs: A (structure of) input tensors.
          state: A (structure of) state tensors and TensorArrays.
          name: Name scope for any created operations.

        Returns:
          `(outputs, next_state, next_inputs, finished)`.
        """
        batch_size = self._batch_size
        beam_width = self._beam_width
        end_token = self._end_token
        length_penalty_weight = self._length_penalty_weight

        with ops.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
            cell_state = state.cell_state
            inputs = nest.map_structure(
                lambda inp: self._merge_batch_beams(inp, s=inp.shape[2:]), inputs)
            cell_state = nest.map_structure(self._maybe_merge_batch_beams, cell_state,
                                            self._cell.state_size)
            cell_outputs, next_cell_state = self._cell(inputs, cell_state)

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            if self.hie:
                cell_outputs = self._mask_outputs_by_lable(cell_outputs, cell_state.last_choice)

                # cell_state.last_choice shape = [batch_size*beam_width,]

            cell_outputs = nest.map_structure(
                lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)

            next_cell_state = nest.map_structure(
                self._maybe_split_batch_beams, next_cell_state, self._cell.state_size)

            beam_search_output, beam_search_state = _beam_search_step(
                time=time,
                logits=cell_outputs,
                next_cell_state=next_cell_state,
                beam_state=state,
                batch_size=batch_size,
                beam_width=beam_width,
                end_token=end_token,
                length_penalty_weight=length_penalty_weight)

            finished = beam_search_state.finished

            sample_ids = beam_search_output.predicted_ids
            # replace the father ids
            if self.hie:
                next_cell_state = beam_search_state.cell_state
                next_cell_state = next_cell_state._replace(last_choice=sample_ids)
                beam_search_state = beam_search_state._replace(cell_state=next_cell_state)

            # sample_ids shape = [batch_size, beam_width]
            next_inputs = control_flow_ops.cond(
                math_ops.reduce_all(finished), lambda: self._start_inputs,
                lambda: self._embedding_fn(sample_ids))

        return (beam_search_output, beam_search_state, next_inputs, finished)

    # added by acharkq
    def _mask_outputs_by_lable(self, outputs, last_choice):
        """outputs shape=[batch_size, num_classes]"""

        vocab_size = array_ops.shape(outputs)[1]
        next_choies = gen_array_ops.gather_v2(params=self.lookup_table, indices=last_choice, axis=0)

        '''get the [batch_size, vocab_size] mask'''
        mask = math_ops.reduce_sum(array_ops.one_hot(indices=next_choies, depth=vocab_size,
                                                     dtype=dtypes.int32), axis=1)
        mask = math_ops.cast(mask, dtype=dtypes.bool)
        # shape = [batch_size, beam_width, vacab_size]
        finished_probs = array_ops.fill(dims=array_ops.shape(outputs), value=outputs.dtype.min)
        return array_ops.where(mask, outputs, finished_probs)


class MyBeamSearchDecoder_v2(MyBeamSearchDecoder):
    '''
    use the leaf attention
    '''

    def __init__(self,
                 cell,
                 embedding,
                 start_tokens,
                 end_token,
                 initial_state,
                 beam_width,
                 lookup_table,
                 step_method,
                 output_layer=None,
                 hie=True,
                 length_penalty_weight=0.0):
        self._step_method = step_method
        super(MyBeamSearchDecoder_v2, self).__init__(cell, embedding, start_tokens, end_token, initial_state,
                                                     beam_width, lookup_table, output_layer, hie, length_penalty_weight)

    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.

        Args:
          time: scalar `int32` tensor.
          inputs: A (structure of) input tensors.
          state: A (structure of) state tensors and TensorArrays.
          name: Name scope for any created operations.

        Returns:
          `(outputs, next_state, next_inputs, finished)`.
        """
        batch_size = self._batch_size
        beam_width = self._beam_width
        end_token = self._end_token
        length_penalty_weight = self._length_penalty_weight

        with ops.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
            cell_state = state.cell_state
            inputs = nest.map_structure(
                lambda inp: self._merge_batch_beams(inp, s=inp.shape[2:]), inputs)
            cell_state = nest.map_structure(self._maybe_merge_batch_beams, cell_state,
                                            self._cell.state_size)
            cell_outputs, next_cell_state = self._cell(inputs, cell_state)

            # finished = tf.Print(state.finished, [state.finished, 'finished', time], summarize=100)
            # not_finished = tf.Print(not_finished, [not_finished, 'not_finished', time], summarize=100)
            # cell_state.last_choice shape = [batch_size * beam_width]
            next_choices = gen_array_ops.gather_v2(self.lookup_table, cell_state.last_choice, axis=0)
            not_finished = tf.not_equal(next_choices[:, 0], end_token)
            next_next_choices = gen_array_ops.gather_v2(self.lookup_table, next_choices[:, 0], axis=0)
            will_finish = tf.logical_and(not_finished, tf.equal(next_next_choices[:, 0], end_token))
            def move(will_finish, last_choice, cell_outputs):
                # cell_outputs = tf.Print(cell_outputs, [cell_outputs, 'cell_outputs', time], summarize=1000)
                # will_finish = tf.Print(will_finish, [will_finish, 'will_finish', time], summarize=100)
                attention_score = self._step_method(last_choice)
                attention_score = attention_score + cell_outputs
                # final = tf.Print(final, [final, 'finalll', time], summarize=1000)
                return tf.where(will_finish, attention_score, cell_outputs)

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)
                # will_finish = tf.Print(will_finish, [will_finish, 'will_finish, beam_search', time], summarize=100)
                cell_outputs = tf.cond(tf.reduce_any(will_finish), false_fn=lambda: cell_outputs,
                                       true_fn=lambda: move(will_finish, cell_state.last_choice, cell_outputs))

            if self.hie:
                cell_outputs = self._mask_outputs_by_lable(cell_outputs, cell_state.last_choice)

                # cell_state.last_choice shape = [batch_size*beam_width,]

            cell_outputs = nest.map_structure(
                lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)

            next_cell_state = nest.map_structure(
                self._maybe_split_batch_beams, next_cell_state, self._cell.state_size)

            beam_search_output, beam_search_state = _beam_search_step(
                time=time,
                logits=cell_outputs,
                next_cell_state=next_cell_state,
                beam_state=state,
                batch_size=batch_size,
                beam_width=beam_width,
                end_token=end_token,
                length_penalty_weight=length_penalty_weight)

            finished = beam_search_state.finished

            # replace the father ids
            sample_ids = beam_search_output.predicted_ids
            next_cell_state = beam_search_state.cell_state
            next_cell_state = next_cell_state._replace(last_choice=sample_ids)
            beam_search_state = beam_search_state._replace(cell_state=next_cell_state)

            # sample_ids shape = [batch_size, beam_width]
            next_inputs = control_flow_ops.cond(
                math_ops.reduce_all(finished), lambda: self._start_inputs,
                lambda: self._embedding_fn(sample_ids))

        return (beam_search_output, beam_search_state, next_inputs, finished)

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A class of Decoders that may sample to generate the next input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib.seq2seq import BasicDecoder, BasicDecoderOutput

__all__ = [
    "MyBasicDecoder",
]

class MyBasicDecoder(BasicDecoder):
    """Basic sampling decoder."""

    # edit by acharkq
    def __init__(self, cell, helper, initial_state, lookup_table, output_layer=None, hie=True):
        """Initialize BasicDecoder.

        Args:
          cell: An `RNNCell` instance.
          helper: A `Helper` instance.
          initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
            The initial state of the RNNCell.
          output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
            `tf.layers.Dense`. Optional layer to apply to the RNN output prior
            to storing the result or sampling.

        Raises:
          TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
        """
        self.lookup_table = lookup_table
        self.hie = hie
        super(MyBasicDecoder, self).__init__(cell=cell, helper=helper, initial_state=initial_state,
                                             output_layer=output_layer)

    # edit by acharkq
    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.

        Args:
          time: scalar `int32` tensor.
          inputs: A (structure of) input tensors.
          state: A (structure of) state tensors and TensorArrays. which contain the choice from the previous step
          name: Name scope for any created operations.

        Returns:
          `(outputs, next_state, next_inputs, finished)`.
        """

        with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
            # will_finished, will finishe after this step, shape = [batch_size,]

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)
            if self.hie:
                cell_outputs = self._mask_outputs_by_lable(cell_outputs, cell_state.last_choice)
                # cell_state.last_choice shape = [batch_size]
                # next_choices shape = [batch_size, max_choices_num]

            sample_ids = self._helper.sample(
                time=time, outputs=cell_outputs, state=cell_state)
            (finished, next_inputs, next_father, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=sample_ids)
            if self.hie:
                next_state = next_state._replace(last_choice=next_father)
        outputs = BasicDecoderOutput(cell_outputs, sample_ids)
        return (outputs, next_state, next_inputs, finished)

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
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
"""A library of helpers for use with SamplingDecoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.contrib.seq2seq.python.ops.helper import TrainingHelper, _unstack_ta, _transpose_batch_time
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


__all__ = [
    "MyTrainingHelper",
]

class MyTrainingHelper(TrainingHelper):
    """A helper for use during training.  Only reads inputs.

    Returned sample_ids are the argmax of the RNN output logits.
    """

    # edited by acharkq
    def __init__(self, inputs, cause_sequence, sequence_length, time_major=False, name=None):
        """
        cause_sequence: shape = [batch_sizze, time_step]
        """
        if not time_major:
            cause_sequence = nest.map_structure(_transpose_batch_time, cause_sequence)
        self._cause_sequnce_tas = nest.map_structure(_unstack_ta, cause_sequence)
        # cause I want the children to know about their father
        nest.map_structure(lambda inp: inp.read(0), self._cause_sequnce_tas)
        super(MyTrainingHelper, self).__init__(inputs, sequence_length, time_major, name)

    # edited by acharkq
    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
        """next_inputs_fn for TrainingHelper."""
        with ops.name_scope(name, "TrainingHelperNextInputs",
                            [time, outputs, state]):
            next_time = time + 1
            finished = (next_time >= self._sequence_length)
            all_finished = math_ops.reduce_all(finished)
            def read_from_ta(inp):
                return inp.read(next_time)
            next_inputs = control_flow_ops.cond(
                all_finished, lambda: self._zero_inputs,
                lambda: nest.map_structure(read_from_ta, self._input_tas))
            next_father = nest.map_structure(read_from_ta, self._cause_sequnce_tas)
            # next_father = tf.Print(next_father, [next_father, "a"], summarize=24)
            return (finished, next_inputs, next_father, state)

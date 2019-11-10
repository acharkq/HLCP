import collections
from tensorflow.python.ops import array_ops, nn_ops, math_ops
from tensorflow.contrib.rnn import LayerNormBasicLSTMCell
from tensorflow.python.framework import tensor_shape


class LSTMStateTuple(collections.namedtuple("LSTMStateTuple", ("c", "h", "last_choice"))):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  and `h` is the output.

  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype


class MyLayerNormBasicLSTMCell(LayerNormBasicLSTMCell):

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units, tensor_shape.TensorShape([]))

    def call(self, inputs, state):
        """LSTM cell with layer normalization and recurrent dropout."""
        c, h, last_choice = state
        args = array_ops.concat([inputs, h], 1)
        concat = self._linear(args)
        dtype = args.dtype

        i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
        if self._layer_norm:
            i = self._norm(i, "input", dtype=dtype)
            j = self._norm(j, "transform", dtype=dtype)
            f = self._norm(f, "forget", dtype=dtype)
            o = self._norm(o, "output", dtype=dtype)

        g = self._activation(j)
        if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
            g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

        new_c = (
                c * math_ops.sigmoid(f + self._forget_bias) + math_ops.sigmoid(i) * g)
        if self._layer_norm:
            new_c = self._norm(new_c, "state", dtype=dtype)
        new_h = self._activation(new_c) * math_ops.sigmoid(o)

        new_state = LSTMStateTuple(new_c, new_h, last_choice)
        return new_h, new_state
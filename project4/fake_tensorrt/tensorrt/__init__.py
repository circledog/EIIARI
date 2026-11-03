"""Offline stand-in for the TensorRT Python API, used by ``check_trt_offline.py``.

No GPU is needed. torch2trt builds its network through this module, exactly
as it would through the real TensorRT: every ``network.add_*`` call becomes a
layer object here. Each layer can then be *evaluated* with PyTorch
operations that follow TensorRT's documented semantics (implicit batch:
axes exclude the batch dimension; explicit batch: they include it), so the
network torch2trt built can be compared with the original model.

What this catches: missing converters, converters that crash, wrong axes /
shapes / broadcasting (TensorRT would refuse to build), and values that
torch2trt silently froze into constants (detected by evaluating with a
different input than the one used for conversion).
What it does not catch: TensorRT engine-builder limits on the Nano (memory,
unsupported tactics) and FP16 rounding (evaluation is FP32).

Never put this folder on ``sys.path`` on the robot: it would hide the real
``tensorrt``. Only ``check_trt_offline.py`` adds it, and only on the PC.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

# TensorRT version reported to torch2trt (it enables converters by version)
__version__ = os.environ.get("FAKE_TRT_VERSION", "7.1.3")
# Networks handed to a builder's build_* call, newest last
CAPTURED = []
# Values fed to network inputs, in add_input order (set by check_trt_offline.py)
FEED = []
# Bumped on every layer attribute change or new feed, so cached values are recomputed
_GEN = [0]


# Error TensorRT itself would raise (invalid layer configuration)
class TRTBuildError(RuntimeError):
    # Plain RuntimeError with its own name
    pass


# Metaclass that creates any missing attribute as a new permissive class (enums, flags, ...)
class _AutoMeta(type):
    # trt.Logger.ERROR, trt.ActivationType.RELU, ... -> cached sub-classes with a .name
    def __getattr__(cls, name):
        # Python's own protocol lookups must fail normally
        if name.startswith("__"):
            # Not one of ours
            raise AttributeError(name)
        # New permissive class standing for this enum value / type
        sub = _AutoMeta(name, (_Auto,), {"name": name})
        # Cache it so the same value always compares equal
        type.__setattr__(cls, name, sub)
        # Hand it out
        return sub

    # int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) -> 0, so 1 << 0 = explicit flag
    def __int__(cls):
        # Every flag maps to bit 0
        return 0


# Object that accepts every attribute, call and context-manager use (logger, config, profile, ...)
class _Auto(metaclass=_AutoMeta):
    # Ignore all constructor arguments
    def __init__(self, *args, **kwargs):
        # Nothing to store
        pass

    # Unknown attribute: another permissive object
    def __getattr__(self, name):
        # Python's own protocol lookups must fail normally
        if name.startswith("__"):
            # Not one of ours
            raise AttributeError(name)
        # Permissive placeholder
        return _Auto()

    # Calling it returns another placeholder
    def __call__(self, *args, **kwargs):
        # Permissive placeholder
        return _Auto()

    # Usable in "with" statements
    def __enter__(self):
        # The object itself
        return self

    # Do not swallow exceptions
    def __exit__(self, *exc):
        # Propagate
        return False


# Any other module attribute (DataType, float32, Logger, IInt8EntropyCalibrator2, ...) is permissive
def __getattr__(name):
    # Python's own protocol lookups must fail normally
    if name.startswith("__"):
        # Not one of ours
        raise AttributeError(name)
    # New permissive class
    value = _AutoMeta(name, (_Auto,), {"name": name})
    # Cache it at module level
    globals()[name] = value
    # Hand it out
    return value


# Layer types (layer.type.name is used by torch2trt to name layers)
LayerType = _AutoMeta("LayerType", (_Auto,), {"name": "LayerType"})


# trt.Weights(dtype) (empty) or trt.Weights(array)
class Weights(object):
    # Keep the array if one was given
    def __init__(self, a=None, *args):
        # Only numpy arrays carry values; a dtype means "empty"
        self.array = a if isinstance(a, np.ndarray) else None


# Torch float32 tensor from numpy / torch / trt.Weights, or None for empty weights
def _arr(w):
    # Empty or missing weights
    if w is None or (isinstance(w, Weights) and w.array is None):
        # Nothing
        return None
    # Unwrap trt.Weights
    if isinstance(w, Weights):
        # Its array
        w = w.array
    # Torch tensors are used directly
    if isinstance(w, torch.Tensor):
        # FP32 copy
        return w.detach().float().cpu()
    # numpy array or scalar
    return torch.from_numpy(np.ascontiguousarray(np.asarray(w, dtype=np.float32)))


# Single set bit of an axes mask -> axis index
def _one_axis(mask):
    # Bits that are set
    axes = [i for i in range(32) if (int(mask) >> i) & 1]
    # Softmax / concatenation work on exactly one axis
    if len(axes) != 1:
        # TensorRT rejects other masks
        raise TRTBuildError("axes mask %r must have exactly one bit set" % mask)
    # The axis
    return axes[0]


# Network tensor whose value is computed on demand from its producing layer
class ITensor(object):
    # Remember the network and how to compute the value
    def __init__(self, net, fn, name=""):
        # Owning network (decides implicit / explicit batch)
        self._net = net
        # Function returning the value (torch tensor with batch dimension)
        self._fn = fn
        # (generation, value) of the last evaluation
        self._cache = None
        # Tensor name
        self.name = name

    # Current value, recomputed after any layer attribute change
    def value(self):
        # Cache missing or outdated
        if self._cache is None or self._cache[0] != _GEN[0]:
            # Recompute
            self._cache = (_GEN[0], self._fn())
        # Cached value
        return self._cache[1]

    # TensorRT shape: without the batch dimension in implicit-batch networks
    @property
    def shape(self):
        # Drop the batch dimension when it is implicit
        return tuple(self.value().shape[self._net.off:])


# One network layer; attribute changes invalidate cached values
class ILayer(object):
    # Store kind, inputs, attributes and the compute function
    def __init__(self, net, kind, inputs, compute, **attrs):
        # Layer attributes are stored without triggering invalidation
        object.__setattr__(self, "_attrs", dict(attrs))
        # Network and inputs
        object.__setattr__(self, "_net", net)
        # Input tensors (ITensor)
        object.__setattr__(self, "_inputs", list(inputs))
        # Layer type (layer.type.name is used by torch2trt for naming)
        object.__setattr__(self, "type", getattr(LayerType, kind))
        # Layer name
        object.__setattr__(self, "name", kind)
        # Output tensor computed by this layer
        object.__setattr__(self, "_outputs", [ITensor(net, lambda: compute(self))])
        # Register the layer with its network
        net._layers.append(self)

    # layer.stride = ..., layer.axis = ..., ... (the *_nd names map to the same attribute)
    def __setattr__(self, key, value):
        # Plain names never touch the cache
        if key in ("name", "precision"):
            # Store directly
            object.__setattr__(self, key, value)
            # Done
            return
        # stride_nd -> stride, padding_nd -> padding, kernel_size_nd -> kernel_size, ...
        self._attrs[key[:-3] if key.endswith("_nd") else key] = value
        # Values depending on this layer must be recomputed
        _GEN[0] += 1

    # Read an attribute set before (or its default)
    def __getattr__(self, key):
        # Internal names are real attributes
        if key.startswith("_"):
            # Not found
            raise AttributeError(key)
        # Attribute dictionary
        attrs = object.__getattribute__(self, "_attrs")
        # Same *_nd mapping as when setting
        key = key[:-3] if key.endswith("_nd") else key
        # Known attribute
        if key in attrs:
            # Its value
            return attrs[key]
        # Unknown: the converter uses something this emulator does not model
        raise AttributeError("offline TensorRT: layer attribute %r is not modelled" % key)

    # Number of outputs (always one here)
    @property
    def num_outputs(self):
        # Single-output layers only
        return 1

    # Output tensor i
    def get_output(self, i):
        # Only index 0 exists
        return self._outputs[i]

    # Input tensor i
    def get_input(self, i):
        # From the stored list
        return self._inputs[i]

    # Replace input tensor i
    def set_input(self, i, tensor):
        # Store it
        self._inputs[i] = tensor
        # Recompute dependent values
        _GEN[0] += 1


# Value of input i of a layer
def _in(layer, i=0):
    # Evaluate the input tensor
    return layer._inputs[i].value()


# Network definition with the add_* methods torch2trt uses
class INetworkDefinition(object):
    # Implicit (TensorRT 7 style) or explicit batch
    def __init__(self, explicit):
        # Explicit-batch networks include the batch dimension in every shape
        self.explicit = explicit
        # Offset between TensorRT axes and torch dimensions
        self.off = 0 if explicit else 1
        # Layers, inputs and outputs
        self._layers, self._inputs, self._outputs = [], [], []

    # Unknown network methods are reported, not silently accepted
    def __getattr__(self, name):
        # Internal names are real attributes
        if name.startswith("_"):
            # Not found
            raise AttributeError(name)
        # Tell which TensorRT call would be needed
        raise AttributeError("offline TensorRT: network.%s is not modelled" % name)

    # torch dimension of TensorRT axis a
    def dim(self, a):
        # Shift by the batch dimension when it is implicit
        return a + self.off

    # ---- inputs / outputs ---------------------------------------------------------- #
    # Network input fed from FEED
    def add_input(self, name, dtype=None, shape=None):
        # Position of this input
        index = len(self._inputs)
        # Value of this input from the feed list
        def value():
            # Fed tensor (FP32)
            v = FEED[index].float()
            # Shape torch2trt declared, with the batch dimension when it is implicit
            expected = tuple(shape) if self.explicit else (v.shape[0],) + tuple(shape)
            # Mismatch means the converter declared a wrong input shape
            if tuple(v.shape) != expected:
                # TensorRT would reject the input
                raise TRTBuildError("input %s declared as %s, fed %s" % (name, expected, tuple(v.shape)))
            # The fed value
            return v
        # New input tensor
        t = ITensor(self, value, name)
        # Remember it
        self._inputs.append(t)
        # Hand it out
        return t

    # Mark a tensor as network output
    def mark_output(self, tensor):
        # Remember it
        self._outputs.append(tensor)

    # Number of layers
    @property
    def num_layers(self):
        # Count
        return len(self._layers)

    # Layer i
    def get_layer(self, i):
        # From the list
        return self._layers[i]

    # Number of inputs
    @property
    def num_inputs(self):
        # Count
        return len(self._inputs)

    # Input i
    def get_input(self, i):
        # From the list
        return self._inputs[i]

    # Number of outputs
    @property
    def num_outputs(self):
        # Count
        return len(self._outputs)

    # Output i
    def get_output(self, i):
        # From the list
        return self._outputs[i]

    # ---- layers -------------------------------------------------------------------- #
    # Constant tensor (broadcast over the batch in implicit-batch networks)
    def add_constant(self, shape, weights):
        # Values in the requested shape
        v = _arr(weights).reshape(tuple(shape))
        # Implicit batch: add a batch dimension of 1 that broadcasts
        v = v if self.explicit else v.unsqueeze(0)
        # Layer returning the constant
        return ILayer(self, "CONSTANT", [], lambda l: v)

    # Identity (also used for dtype casts)
    def add_identity(self, input):
        # Layer passing the value through
        return ILayer(self, "IDENTITY", [input], lambda l: _in(l))

    # 2-D convolution
    def add_convolution(self, input, num_output_maps, kernel_shape, kernel, bias=None):
        # Weight tensor, its kernel size and the optional bias
        w, b, k = _arr(kernel), _arr(bias), tuple(kernel_shape)

        # Evaluate with the current stride / padding / groups
        def compute(l):
            # Input value
            x = _in(l)
            # Weights in (out, in / groups, kh, kw) layout
            weight = w.reshape(num_output_maps, -1, k[0], k[1])
            # Input channels must match the weights
            if weight.shape[1] * l.num_groups != x.shape[1]:
                # TensorRT would reject the layer
                raise TRTBuildError("convolution expects %d input channels, got %d"
                                    % (weight.shape[1] * l.num_groups, x.shape[1]))
            # Convolution with TensorRT's symmetric padding
            return F.conv2d(x, weight, b, tuple(l.stride), tuple(l.padding), tuple(l.dilation), l.num_groups)
        # Layer with TensorRT's default attributes
        return ILayer(self, "CONVOLUTION", [input], compute, stride=(1, 1), padding=(0, 0),
                      dilation=(1, 1), num_groups=1, kernel_size=k)

    # Same layer, newer API name
    add_convolution_nd = add_convolution

    # Per-tensor / per-channel / per-element (x * scale + shift) ** power (BatchNorm)
    def add_scale(self, input, mode, shift=None, scale=None, power=None, channel_axis=None):
        # Weights (None = default)
        sh, sc, pw = _arr(shift), _arr(scale), _arr(power)

        # Evaluate
        def compute(l):
            # Input value
            x = _in(l)
            # Channel dimension: given axis, else the first non-batch axis
            c = self.dim(channel_axis) if channel_axis is not None else 1
            # Shape that broadcasts a per-channel vector along dimension c
            per_channel = [1] * x.dim()
            # Channels along c
            per_channel[c] = -1

            # Reshape one weight to the mode's layout
            def shaped(t):
                # Missing weights keep their default
                if t is None:
                    # Caller substitutes the default
                    return None
                # One value for the whole tensor
                if mode.name == "UNIFORM":
                    # Scalar
                    return t.reshape(())
                # One value per channel
                if mode.name == "CHANNEL":
                    # Broadcast along c
                    return t.reshape(per_channel)
                # One value per element
                return t.reshape((1,) + tuple(x.shape[1:]))
            # Apply scale, shift and power in TensorRT's order
            y = x * shaped(sc) if sc is not None else x
            # Shift
            y = y + shaped(sh) if sh is not None else y
            # Power
            return y ** shaped(pw) if pw is not None else y
        # Layer
        return ILayer(self, "SCALE", [input], compute)

    # Same layer with an explicit channel axis
    def add_scale_nd(self, input, mode, shift=None, scale=None, power=None, channel_axis=0):
        # Reuse add_scale
        return self.add_scale(input, mode, shift, scale, power, channel_axis)

    # Activation functions
    def add_activation(self, input, type):
        # Evaluate with the current alpha / beta
        def compute(l):
            # Input value
            x = _in(l)
            # Name of the activation
            n = type.name
            # ReLU
            if n == "RELU":
                # max(x, 0)
                return torch.relu(x)
            # Sigmoid
            if n == "SIGMOID":
                # 1 / (1 + e^-x)
                return torch.sigmoid(x)
            # tanh
            if n == "TANH":
                # tanh
                return torch.tanh(x)
            # Clip to [alpha, beta] (ReLU6)
            if n == "CLIP":
                # Clamp
                return torch.clamp(x, l.alpha, l.beta)
            # Not modelled
            raise TRTBuildError("activation %s is not modelled" % n)
        # Layer
        return ILayer(self, "ACTIVATION", [input], compute, alpha=0.0, beta=0.0)

    # Max / average pooling
    def add_pooling(self, input, type, window_size):
        # Window size as a tuple
        win = tuple(window_size)

        # Evaluate with the current stride / padding
        def compute(l):
            # Input value
            x = _in(l)
            # Max pooling
            if type.name == "MAX":
                # Padded positions never win
                return F.max_pool2d(x, win, tuple(l.stride), tuple(l.padding))
            # Average pooling; TensorRT excludes padding from the count by default
            return F.avg_pool2d(x, win, tuple(l.stride), tuple(l.padding),
                                count_include_pad=not l.average_count_excludes_padding)
        # Layer with TensorRT's default attributes
        return ILayer(self, "POOLING", [input], compute, stride=(1, 1), padding=(0, 0),
                      average_count_excludes_padding=True, window_size=win)

    # Same layer, newer API name
    add_pooling_nd = add_pooling

    # Element-wise operation of two tensors with TensorRT's broadcasting rules
    def add_elementwise(self, input1, input2, op):
        # Evaluate
        def compute(l):
            # Both values
            a, b = _in(l, 0), _in(l, 1)
            # TensorRT needs the same number of dimensions
            if a.dim() != b.dim():
                # Build error
                raise TRTBuildError("elementwise inputs have %d and %d dimensions" % (a.dim(), b.dim()))
            # Every dimension must match or be 1
            for da, db in zip(a.shape, b.shape):
                # Incompatible sizes
                if da != db and 1 not in (da, db):
                    # Build error
                    raise TRTBuildError("elementwise shapes %s and %s do not broadcast"
                                        % (tuple(a.shape), tuple(b.shape)))
            # Operation table
            ops = {"SUM": torch.add, "PROD": torch.mul, "SUB": torch.sub, "DIV": torch.div,
                   "MAX": torch.maximum, "MIN": torch.minimum, "POW": torch.pow}
            # Unknown operation
            if op.name not in ops:
                # Not modelled
                raise TRTBuildError("elementwise %s is not modelled" % op.name)
            # Apply it
            return ops[op.name](a, b)
        # Layer
        return ILayer(self, "ELEMENTWISE", [input1, input2], compute)

    # Transpose / reshape / transpose
    def add_shuffle(self, input):
        # Evaluate with the current permutations and shape
        def compute(l):
            # Input value
            x = _in(l)
            # Dimensions TensorRT may touch (not the implicit batch)
            off = self.off
            # First permutation
            if l.first_transpose is not None:
                # Batch stays in front
                x = x.permute(list(range(off)) + [off + int(p) for p in l.first_transpose])
            # Reshape
            if l.reshape_dims is not None:
                # Current TensorRT shape
                cur = list(x.shape[off:])
                # 0 copies the input dimension, -1 is inferred
                dims = [cur[i] if int(d) == 0 else int(d) for i, d in enumerate(l.reshape_dims)]
                # Element counts must agree
                try:
                    # New shape, batch in front
                    x = x.reshape(list(x.shape[:off]) + dims)
                # Impossible reshape
                except RuntimeError as e:
                    # Build error
                    raise TRTBuildError("shuffle reshape %s -> %s: %s" % (cur, dims, e))
            # Second permutation
            if l.second_transpose is not None:
                # Batch stays in front
                x = x.permute(list(range(off)) + [off + int(p) for p in l.second_transpose])
            # Result
            return x
        # Layer with no permutation or reshape
        return ILayer(self, "SHUFFLE", [input], compute, first_transpose=None, reshape_dims=None,
                      second_transpose=None)

    # Concatenation along one axis
    def add_concatenation(self, inputs):
        # Evaluate with the current axis
        def compute(l):
            # All values
            vs = [t.value() for t in l._inputs]
            # Number of TensorRT axes
            nd = vs[0].dim() - self.off
            # TensorRT's default axis: number of axes minus three, at least 0
            axis = l.axis if l.axis is not None else max(0, nd - 3)
            # torch dimension of that axis
            d = self.dim(axis)
            # Every input needs the same rank and equal sizes outside the axis
            for v in vs:
                # Rank or size mismatch
                if v.dim() != vs[0].dim() or any(v.shape[i] != vs[0].shape[i]
                                                 for i in range(v.dim()) if i != d):
                    # Build error
                    raise TRTBuildError("concatenation inputs %s do not fit along axis %d"
                                        % ([tuple(t.shape[self.off:]) for t in vs], axis))
            # Concatenate
            return torch.cat(vs, d)
        # Layer with the default axis
        return ILayer(self, "CONCATENATION", list(inputs), compute, axis=None)

    # Fully connected layer on the last three dimensions
    def add_fully_connected(self, input, num_outputs, kernel, bias=None):
        # Weights and bias
        w, b = _arr(kernel), _arr(bias)

        # Evaluate
        def compute(l):
            # Input value
            x = _in(l)
            # TensorRT needs at least three non-batch dimensions (C, H, W)
            if x.dim() - self.off < 3:
                # Build error
                raise TRTBuildError("fully connected input needs >= 3 dimensions, got %s"
                                    % (tuple(x.shape[self.off:]),))
            # Leading dimensions are kept
            lead = list(x.shape[:-3])
            # Flatten C, H, W
            flat = x.reshape(lead + [-1])
            # Matrix product with the (outputs, inputs) weights
            y = F.linear(flat, w.reshape(num_outputs, -1), b)
            # Output shape (..., K, 1, 1)
            return y.reshape(lead + [num_outputs, 1, 1])
        # Layer
        return ILayer(self, "FULLY_CONNECTED", [input], compute)

    # Reduction over the axes in a bit mask
    def add_reduce(self, input, op, axes, keep_dims):
        # Evaluate
        def compute(l):
            # Input value
            x = _in(l)
            # torch dimensions of the set bits
            dims = [self.dim(i) for i in range(32) if (int(axes) >> i) & 1]
            # Operation table
            ops = {"SUM": torch.sum, "AVG": torch.mean, "PROD": None, "MAX": torch.amax, "MIN": torch.amin}
            # Product over several axes
            if op.name == "PROD":
                # Reduce one axis at a time, highest first
                for d in sorted(dims, reverse=True):
                    # Product along d
                    x = torch.prod(x, d, keepdim=bool(keep_dims))
                # Result
                return x
            # Other operations
            return ops[op.name](x, dim=dims, keepdim=bool(keep_dims))
        # Layer
        return ILayer(self, "REDUCE", [input], compute)

    # Softmax along one axis
    def add_softmax(self, input):
        # Evaluate with the current axes
        def compute(l):
            # Input value
            x = _in(l)
            # TensorRT's default axis: number of axes minus three, at least 0
            axis = _one_axis(l.axes) if l.axes is not None else max(0, x.dim() - self.off - 3)
            # Softmax along that axis
            return torch.softmax(x, self.dim(axis))
        # Layer with the default axes
        return ILayer(self, "SOFTMAX", [input], compute, axes=None)

    # Strided slice
    def add_slice(self, input, start, shape, stride):
        # Evaluate with the current start / shape / stride
        def compute(l):
            # Input value
            x = _in(l)
            # Slice every TensorRT axis
            for a, (s0, n, st) in enumerate(zip(l.start, l.shape, l.stride)):
                # Indices taken along this axis
                idx = torch.arange(int(n)) * int(st) + int(s0)
                # Out of range
                if len(idx) and (idx.max() >= x.shape[self.dim(a)] or idx.min() < 0):
                    # Build error
                    raise TRTBuildError("slice out of range on axis %d" % a)
                # Take them
                x = x.index_select(self.dim(a), idx)
            # Result
            return x
        # Layer
        return ILayer(self, "SLICE", [input], compute, start=tuple(start), shape=tuple(shape),
                      stride=tuple(stride))

    # Unary operation
    def add_unary(self, input, op):
        # Operation table
        ops = {"EXP": torch.exp, "LOG": torch.log, "SQRT": torch.sqrt, "RECIP": torch.reciprocal,
               "ABS": torch.abs, "NEG": torch.neg, "SIN": torch.sin, "COS": torch.cos, "FLOOR": torch.floor,
               "CEIL": torch.ceil}
        # Unknown operation
        if op.name not in ops:
            # Not modelled
            raise TRTBuildError("unary %s is not modelled" % op.name)
        # Layer applying it
        return ILayer(self, "UNARY", [input], lambda l: ops[op.name](_in(l)))

    # Matrix product of the last two dimensions
    def add_matrix_multiply(self, input0, op0, input1, op1):
        # Evaluate
        def compute(l):
            # Both values
            a, b = _in(l, 0), _in(l, 1)
            # Optional transposes
            a = a.transpose(-1, -2) if op0.name == "TRANSPOSE" else a
            # Second operand
            b = b.transpose(-1, -2) if op1.name == "TRANSPOSE" else b
            # Vector operands are not modelled
            if "VECTOR" in (op0.name, op1.name):
                # Not modelled
                raise TRTBuildError("matrix multiply VECTOR is not modelled")
            # Product
            return torch.matmul(a, b)
        # Layer
        return ILayer(self, "MATRIX_MULTIPLY", [input0, input1], compute)

    # Zero padding of the last two dimensions
    def add_padding(self, input, pre_padding, post_padding):
        # Evaluate
        def compute(l):
            # Padding (top, left) and (bottom, right)
            (t, le), (bo, r) = tuple(l.pre_padding), tuple(l.post_padding)
            # Zero padding
            return F.pad(_in(l), (le, r, t, bo))
        # Layer
        return ILayer(self, "PADDING", [input], compute, pre_padding=tuple(pre_padding),
                      post_padding=tuple(post_padding))

    # Same layer, newer API name
    add_padding_nd = add_padding


# Plugin registry without plugins (old torch2trt lists its own plugins on import)
class _PluginRegistry(object):
    # No plugin creators
    plugin_creator_list = []

    # Registering is accepted and ignored
    def register_creator(self, *args, **kwargs):
        # Nothing to do
        return True


# trt.get_plugin_registry()
def get_plugin_registry():
    # Empty registry
    return _PluginRegistry()


# Builder: creates networks and records the one it is asked to build
class Builder(_Auto):
    # create_network() -> implicit batch, create_network(1 << EXPLICIT_BATCH) -> explicit
    def create_network(self, flags=0):
        # New network definition
        return INetworkDefinition(explicit=bool(flags))

    # TensorRT 7 API
    def build_cuda_engine(self, network):
        # Remember the network for evaluation
        CAPTURED.append(network)
        # Placeholder engine
        return _Auto()

    # TensorRT 7.1 / 8 API
    def build_engine(self, network, config=None):
        # Same as above
        return self.build_cuda_engine(network)

    # TensorRT 8.x API
    def build_serialized_network(self, network, config=None):
        # Same as above
        return self.build_cuda_engine(network)


# Evaluate a captured network on new input values
def evaluate(network, inputs):
    """Outputs of ``network`` (list of torch tensors) for ``inputs`` (list, with batch dim)."""
    # Replace the feed in place (input layers read it by index)
    FEED[:] = list(inputs)
    # Every cached value is now outdated
    _GEN[0] += 1
    # Evaluate all marked outputs
    return [t.value() for t in network._outputs]

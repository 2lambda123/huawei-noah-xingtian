"""Action distribution with mindspore"""
import numpy as np
from xt.model.ms_compat import ms, Cast, ReduceSum, ReduceMax, SoftmaxCrossEntropyWithLogits
from mindspore import ops

class ActionDist:
    """Build base action distribution."""
    def init_by_param(self, param):
        raise NotImplementedError

    def flatparam(self):
        raise NotImplementedError

    def sample(self, repeat):
        """Sample action from this distribution."""
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def get_shape(self):
        return self.flatparam().shape.as_list()

    @property
    def shape(self):
        return self.get_shape()

    def __getitem__(self, idx):
        return self.flatparam()[idx]

    def neglog_prob(self, x):
        raise NotImplementedError

    def log_prob(self, x):
        """Calculate the log-likelihood."""
        return -self.neglog_prob(x)

    def mode(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

class DiagGaussianDist(ActionDist):
    """Build Diagonal Gaussian distribution, each vector represented one distribution."""

    def __init__(self, size):
        self.size = size

    def init_by_param(self, param):
        self.param = param
        self.mean, self.log_std = ops.split(self.param, axis=-1, output_num=2)
        self.std = ops.exp(self.log_std)

    def flatparam(self):
        return self.param

    def sample_dtype(self):
        return ms.float32

    def neglog_prob(self, x):
        reduce_sum = ReduceSum(keep_dims=True)
        return 0.5 * np.log(2.0 * np.pi) * Cast()((ops.shape(x)[-1]), ms.float32) + \
            0.5 * reduce_sum(ops.square((x - self.mean) / self.std), axis=-1) + \
            reduce_sum(self.log_std, axis=-1)

    def mode(self):
        return self.mean

    def entropy(self):
        reduce_sum = ReduceSum(keep_dims=True)
        return reduce_sum(self.log_std + 0.5 * (np.log(2.0 * np.pi) + 1.0), axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianDist), 'Distribution type not match.'
        reduce_sum = ReduceSum(keep_dims=True)
        return reduce_sum(
            (ops.square(self.std) + ops.square(self.mean - other.mean)) / (2.0 * ops.square(other.std)) +
            other.log_std - self.log_std - 0.5,
            axis=-1)

    def sample(self, repeat=None):
        return self.mean + self.std * ops.normal(ops.shape(self.mean), dtype=ms.float32)

class CategoricalDist(ActionDist):

    def __init__(self, size):
        self.size = size

    def init_by_param(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def sample_dtype(self):
        return ms.int32

    def neglog_prob(self, x):
        net = ms.nn.OneHot(axis=-1,depth = self.size)
        x = net(ms.Tensor(x,ms.int32))
        loss = SoftmaxCrossEntropyWithLogits(sparse=False, reduction="mean")
        logits = ms.Tensor(self.logits, ms.float32)
        neglogp = loss(logits, x)
        return ops.expand_dims(neglogp, axis=-1)

    def entropy(self):
        reduce_max = ReduceMax(keep_dims=True)
        rescaled_logits = self.logits - reduce_max(self.logits, axis=-1)
        exp_logits = ops.exp(rescaled_logits)
        reduce_sum = ReduceSum(keep_dims=True)
        z = reduce_sum(exp_logits, axis=-1)
        p = exp_logits / z
        return reduce_sum(p * (ops.log(z) - rescaled_logits), axis=-1)

    def kl(self, other):
        assert isinstance(other, CategoricalDist), 'Distribution type not match.'
        reduce_max = ReduceMax(keep_dims=True)
        reduce_sum = ReduceSum(keep_dims=True)
        rescaled_logits_self = self.logits - reduce_max(self.logits, axis=-1)
        rescaled_logits_other = other.logits - reduce_max(other.logits, axis=-1)
        exp_logits_self = ops.exp(rescaled_logits_self)
        exp_logits_other = ops.exp(rescaled_logits_other)
        z_self = reduce_sum(exp_logits_self, axis=-1)
        z_other = reduce_sum(exp_logits_other, axis=-1)
        p = exp_logits_self / z_self
        return reduce_sum(p * (rescaled_logits_self - ops.log(z_self) - rescaled_logits_other + ops.log(z_other)),
                             axis=-1)

    def sample(self):
        # u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        # return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1, output_type=tf.int32)
        return ops.squeeze(ops.random_categorical(logits=self.logits, num_sample=1, dtype=ms.int32), axis=-1)


def make_dist(ac_type, ac_dim):
    if ac_type == 'Categorical':
        return CategoricalDist(ac_dim)
    elif ac_type == 'DiagGaussian':
        return DiagGaussianDist(ac_dim)
    else:
        raise NotImplementedError
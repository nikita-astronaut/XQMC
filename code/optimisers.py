import numpy as np

class Optimiser(object):
    def __init__(self, opt_parameters):
        self._unwrap_parameters(opt_parameters)
        self._reset_state()
        return

    def _unwrap_parameters(self, opt_parameters):
        raise NotImplementedError

    def get_step(self, gradient):
        raise NotImplementedError

    def _reset_state(self):
        raise NotImplementedError


class AdamOptimiser(Optimiser):
    def __init__(self, opt_parameters):
        super().__init__(opt_parameters)
        return

    def _unwrap_parameters(self, opt_parameters):
        self.beta1 = opt_parameters[0]
        self.beta2 = opt_parameters[1]
        self.epsilon = opt_parameters[2]
        self.lr = opt_parameters[3]
        return

    def _reset_state(self):
        self.vd = 0.0
        self.sd = 0.0
        self.t = 1
        return

    def get_step(self, gradient):
        self.vd = self.beta1 * self.vd + (1 - self.beta1) * gradient
        self.sd = self.beta2 * self.sd + (1 - self.beta2) * gradient ** 2
        self.vd /= (1 - self.beta1 ** self.t)
        self.sd /= (1 - self.beta2 ** self.t)

        nd = self.vd / (np.sqrt(self.sd) + self.epsilon)
        self.t += 1
        return self.lr * nd

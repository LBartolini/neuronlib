import math


class Neurone(object):

    def __init__(self, n_inp, lr=0.01):
        self.weights = self.set_w(n_inp)
        self.bias = 1
        self.lr = lr

    def set_w(self, n_inp):
        x = []
        for i in range(n_inp):
            x.append(1)
        return x

    def set_pa(self, inputs):
        pa = 0
        for i in range(len(inputs)):
            pa += inputs[i] * self.weights[i]
        pa += self.bias
        return pa

    def learn(self, inputs, target, to_print=False):
        out, pa = self.output(inputs)
        diff = out - target
        cost = (out - target) ** 2
        if to_print:
            print('input:', inputs, 'target:', target, 'Predizione:', out, 'Costo:', cost)
        for i, w in enumerate(self.weights):
            self.weights[i] -= self.lr * 2 * diff * inputs[i] * self.sigmoide(pa, deriv=True)
        self.bias -= self.lr * 2 * diff * self.sigmoide(pa, deriv=True)
        return out

    def output(self, inputs):
        pa = self.set_pa(inputs)
        return self.sigmoide(pa), pa

    def sigmoide(self, pa, deriv=False):
        if not deriv:
            return 1 / (1 + math.exp(-pa))
        else:
            return self.sigmoide(pa) * (1 - self.sigmoide(pa))
import random

LAMBDA = 0.8

class LeakyNeuron():
    def __init__(self, threshold=0.6, max_refractory=10):
        self.voltage = 0.
        self.threshold = threshold
        self.spike_voltage = 5. * threshold
        self.refractory_count = 0
        self.max_refractory = max_refractory

    def step(self, input_amount):
        self.voltage = LAMBDA * self.voltage + input_amount
        did_spike = 0.0
        if self.refractory_count > 0:
            self.voltage = 0
            self.refractory_count -= 1
        elif self.voltage > self.threshold:
            self.spike()
            did_spike = 1.0
        return did_spike

    def spike(self):
        self.refractory_count = self.max_refractory
        self.voltage = self.spike_voltage

class ProbabilisticLeakyNeuron():
    def __init__(self, threshold= 1.0):
        self.voltage = 0.
        self.threshold = threshold
        self.spike_voltage = 5. * threshold
        self.refractory_count = 0
        self.delay = 5
        self.delay_count = 0

    def step(self, input_amount):
        self.voltage = LAMBDA * self.voltage + input_amount
        did_spike = 0.0
        threshold = self.threshold
        if self.refractory_count > 0:
            t_mult = 2. * (10 - self.refractory_count) / 10 + 1. * (self.refractory_count / 10)
            threshold *= t_mult
            # self.voltage = 0
            self.refractory_count -= 1
        elif self.voltage > threshold / 2.:
            p = self.voltage - (threshold / 2.) / (threshold / 2)
            r = random.random()
            if r < p:
                self.spike()
                did_spike = 1.0
        return did_spike

    def spike(self):
        if self.refractory_count == 0:
            self.refractory_count = 10
            self.voltage = self.spike_voltage


class LCNeuronPool:
    def __init__(self, n_neurons=250, coupled=False):
        self.neurons = [ProbabilisticLeakyNeuron() for _ in range(n_neurons)]
        self.prev_spikes = [0.0] * n_neurons
        self.input_weight = 1.5
        self.lateral_inhib_weight = -0.01 / 249
        self.coupled = coupled

        if not coupled:
            self.noise_freq = 0.
            self.input_weight = 1.8
        else:
            self.noise_freq = 1 / 1000

    def step(self, input_spike):
        spike_count = 0
        if self.coupled:
            # did_spike = self.neurons[0].step(input_spike * self.input_weight + self.prev_spikes[0] * self.lateral_inhib_weight)
            # self.prev_spikes = [did_spike]
            # return 250 * did_spike
            spikes = []
            for i, neuron in enumerate(self.neurons):
                other_spike_count = sum(self.prev_spikes[:i] + self.prev_spikes[i+1:])
                did_spike = neuron.step(input_spike * self.input_weight + other_spike_count * self.lateral_inhib_weight)
                if did_spike:
                    spike_count += 1
                spikes.append(did_spike)
            self.prev_spikes = spikes
            return spike_count
        else:
            spikes = []
            for i, neuron in enumerate(self.neurons):
                uncouple_noise = 0 # 1.0 * random.random()
                other_spike_count = sum(self.prev_spikes[:i] + self.prev_spikes[i+1:])
                did_spike = neuron.step((uncouple_noise + input_spike) * self.input_weight + other_spike_count * self.lateral_inhib_weight)
                if did_spike:
                    spike_count += 1
                spikes.append(did_spike)
            self.prev_spikes = spikes
            return spike_count

class DecisionLayer:
    def __init__(self):
        self.neurons = (LeakyNeuron(), LeakyNeuron())
        # TODO: mess around with these weights
        self.direct_weight_target = 0.6
        self.indirect_weight_target = 0.2
        self.direct_weight_distractor = 0.6
        self.indirect_weight_distractor = 0.4
        self.recurrent_weight = 0.1
        self.lateral_inhib_weight_left = -.5
        self.lateral_inhib_weight_right = -.7
        self.lc_weight_left = -.1 / 250
        self.lc_weight_right = -.6 / 250
        
    def step(self, target_spike, distractor_spike, prev_dl_spikes, lc_spike_count):
        left_spike = self.neurons[0].step(target_spike * self.direct_weight_target \
                                          + distractor_spike * self.indirect_weight_distractor \
                                          + prev_dl_spikes[0] * self.recurrent_weight \
                                          + prev_dl_spikes[1] * self.lateral_inhib_weight_right \
                                          + lc_spike_count * self.lc_weight_left)
        right_spike = self.neurons[1].step(target_spike * self.indirect_weight_target \
                                          + distractor_spike * self.direct_weight_distractor \
                                          + prev_dl_spikes[1] * self.recurrent_weight \
                                          + prev_dl_spikes[0] * self.lateral_inhib_weight_left \
                                          + lc_spike_count * self.lc_weight_right)
        
        return left_spike, right_spike
        

class ResponseUnit:
    def __init__(self):
        self.neuron = LeakyNeuron(max_refractory=4)
        self.bias_amt = 0.2
        self.lc_inhib_weight = -0.65 / 250
        self.recurrent_weight = 0.1
        self.dl_weight = 1.2
    
    def step(self, lc_spike_count, prev_spike, dl_spike):
        input_voltage = lc_spike_count * self.lc_inhib_weight \
                              + prev_spike * self.recurrent_weight \
                              + dl_spike * self.dl_weight
        # print("Response Input Voltage = ", input_voltage)
        return self.neuron.step(input_voltage)

class Model:
    def __init__(self, coupled=False):
        self.lc_pool = LCNeuronPool(coupled=coupled)
        self.decision_layer = DecisionLayer()
        self.response_unit = ResponseUnit()
        
        self.dl_spikes = (0.0, 0.0)
        self.lc_spike_count = 0.0
        self.response_spike = 0.0

    def step(self, target_spike, distractor_spike):
        self.dl_spikes = self.decision_layer.step(target_spike,
                                                  distractor_spike,
                                                  self.dl_spikes,
                                                  self.lc_spike_count)
        # print("DL Spikes = ", self.dl_spikes)
        self.lc_spike_count = self.lc_pool.step(self.dl_spikes[0])
        # print("LC Spike Count = ", self.lc_spike_count)
        self.response_spike = self.response_unit.step(self.lc_spike_count, self.response_spike, self.dl_spikes[0])
        return self.response_spike
        
if __name__ == '__main__':
    m = Model()
    print(m.step(1.0, 0.0))
    print(m.step(0.0, 0.0))
    print(m.step(0.0, 0.0))
    print(m.step(0.0, 0.0))
    print(m.step(0.0, 0.0))
    print(m.step(0.0, 0.0))
    print(m.step(0.0, 0.0))

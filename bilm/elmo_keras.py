from keras import backend as K
from keras.engine import Layer

class WeightElmo(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        elmo_layers = input_shape[len(input_shape)-1]
        self.W = self.add_weight(name='layer_weights', shape=(elmo_layers,), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(1,), initializer='ones', trainable=True)
        self.norm = K.constant(1.0 / elmo_layers, dtype='float32')
        super(WeightElmo, self).build(input_shape)

    def call(self, x, **kwargs):
        w = K.softmax(self.W + self.norm)
        return K.sum(x * w, axis=3) * self.gamma

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
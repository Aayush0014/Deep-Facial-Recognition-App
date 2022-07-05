# Custom L1 Distance layer module

# Importing dependencies
import tensorflow as tf
from keras.api._v2.keras.layers import Layer


# custom L1 distance layer
class L1Dist(Layer):
    # Inheritance
    def __init__(self,**kwargs):
        super().__init__()
    
    # similarity calculation
    def cell(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

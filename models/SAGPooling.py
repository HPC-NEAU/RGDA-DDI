import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np


class SAGPooling(Layer):
    def __init__(self, **kwargs):
        super(SAGPooling, self).__init__(**kwargs)
        #self.ratio = ratio

    def build(self, input_shape):
        self.fc = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x, adj = inputs

        # Compute node importance scores
        scores = self.fc(x)

        # Apply the graph reduction
        scores = tf.squeeze(scores, axis=-1)
        num_nodes = tf.shape(x)[-1]
        num = tf.shape(x)[-2]
        top_k = tf.cast((num_nodes*num) // num_nodes, dtype=tf.int32)
        _, indices = tf.nn.top_k(scores, k=top_k)
        mask = tf.one_hot(indices, num_nodes)

        adj = tf.matmul(mask, tf.matmul(mask, adj),transpose_a=True)
        x = tf.matmul(mask,tf.matmul(mask,x,transpose_a=True))

        return x+inputs[0], adj

if __name__ == '__main__':
    a = tf.convert_to_tensor(np.ones(1024,1,64))
    b = tf.convert_to_tensor(np.ones(1024,16,64))
    mode = SAGPooling()
    c,_ = mode([a,b])
    print(c.shape)
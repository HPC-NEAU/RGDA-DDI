import numpy as np
import tensorflow as tf

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(SelfAttention, self).__init__()
        self.Q1 = tf.keras.layers.Dense(units)
        self.K1 = tf.keras.layers.Dense(units)
        self.V1 = tf.keras.layers.Dense(1)

        self.Q2 = tf.keras.layers.Dense(units)
        self.K2 = tf.keras.layers.Dense(units)
        self.V2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        #print("duck")
        one = inputs[0]
        two = inputs[1]
        # inputs shape: (batch_size, seq_len, embedding_dim)
        # hidden shape: (batch_size, seq_len, units)
        hidden1 = tf.nn.tanh(self.K1(one) * self.Q2(two))

        # score shape: (batch_size, seq_len, 1)
        score1 = self.V1(hidden1)

        hidden2 = tf.nn.tanh(self.Q2(one) * self.K1(two))

        # score shape: (batch_size, seq_len, 1)
        score2 = self.V2(hidden2)




        attention_weights1 = tf.nn.tanh(score1)# axis=1)

        context_vector1 = attention_weights1 * inputs[0]
        #context_vector1 = tf.reduce_sum(context_vector1, axis=1)

        attention_weights2 = tf.nn.softmax(score2, axis=1)

        context_vector2 = attention_weights2 * inputs[1]
        #context_vector2 = tf.reduce_sum(context_vector2, axis=1)

        return context_vector1,context_vector2

if __name__ == '__main__':
    #(None, 2, 128) (None, 2, 128)
    import numpy
    a = tf.convert_to_tensor(np.ones([1,64]))
    model = SelfAttention()
    b,c = model([a,a])
    print(b.shape,c.shape)

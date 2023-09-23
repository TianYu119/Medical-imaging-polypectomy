import tensorflow as tf 



def convformer(input_tensor, filters, padding="same"):
    x = tf.keras.layers.LayerNormalization()(input_tensor)
    x = tf.keras.layers.SeparableConv2D(filters, kernel_size=(3, 3), padding=padding)(x)

    # MLP层
    mlp_output = MLPBlock(units=filters)(x)

    out = tf.keras.layers.Add()([mlp_output, input_tensor])

    x1 = tf.keras.layers.Dense(filters, activation="gelu")(out)
    x1 = tf.keras.layers.Dense(filters)(x1)
    out_tensor = tf.keras.layers.Add()([out, x1])

    return out_tensor




class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, units, activation="relu",name=None,trainable=True,dtype=None):
        super(MLPBlock, self).__init__(trainable=trainable,name=name,dtype=dtype)
        self.units = units
        self.activation = activation
        self.dense1 = tf.keras.layers.Dense(units, activation=activation)
        self.dense2 = tf.keras.layers.Dense(units, activation=None)  # No activation on the last layer

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': self.activation
        })
        return config


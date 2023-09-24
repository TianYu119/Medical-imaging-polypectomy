import tensorflow as tf 



def convformer(input_tensor, filters, padding="same"):
    x = tf.keras.layers.LayerNormalization()(input_tensor)
    x = tf.keras.layers.SeparableConv2D(filters, kernel_size=(3, 3), padding=padding)(x)

    #空间注意力
    attn_output = SpatialAttentionLayer()(x)
    x = tf.keras.layers.Add()([x, attn_output])

    # MLP层
    mlp_output = MLPBlock(units=filters)(x)
    out = tf.keras.layers.Add()([mlp_output, input_tensor])

    x1 = tf.keras.layers.Dense(filters, activation="relu")(out)
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

class SpatialAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, **kwargs):
        super(SpatialAttentionLayer, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = tf.keras.layers.Conv2D(1, self.kernel_size, padding='same', activation='sigmoid')

    def call(self, inputs):
        # 计算空间注意力图
        attention_map = self.conv1(inputs)
        # 将注意力图应用于输入特征图
        output = inputs * attention_map
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
        })
        return config


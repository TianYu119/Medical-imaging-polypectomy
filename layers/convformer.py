import tensorflow as tf 
from tensorflow.keras.regularizers import l2


def convformer(input_tensor, filters, padding="same"):
    x = tf.keras.layers.LayerNormalization()(input_tensor)
    # input_tensor = tf.keras.layers.BatchNormalization()(input_tensor)
    # x = tf.keras.layers.SeparableConv2D(filters, kernel_size=(3, 3), padding=padding)(x)
    x = tf.keras.layers.SeparableConv2D(filters, kernel_size=(3, 3), padding=padding, 
                                         kernel_regularizer=l2(0.01))(input_tensor)  # 添加正则化
    # x = tf.keras.layers.BatchNormalization()(x)

        # 空间注意力
    spatial_attention = SpatialAttentionLayer()(x)


    # 在此处加入边界聚合模块
    merged_with_boundary_s = boundary_aggregation_module(spatial_attention, filters)

    # 通道注意力
    channel_attention = ChannelAttentionLayer()(x)

    # 在此处加入边界聚合模块
    merged_with_boundary_c = boundary_aggregation_module(channel_attention, filters)


    # 元素级相加合并注意力输出
    merged_attention = tf.keras.layers.Add()([merged_with_boundary_s, merged_with_boundary_c])
  
    # 边界预测
    boundary_feature = boundary_prediction_subnetwork(merged_attention, filters)
    
    # 融合边界特征
    # 这里只是一个简单的融合示例，您可以使用其他方法如Concatenate，Multiply等
    fused = tf.keras.layers.Add()([merged_attention, boundary_feature])
    
    # MLP层
    mlp_output = MLPBlock(units=filters)(fused)
    
    
    # # MLP层
    # mlp_output = MLPBlock(units=filters)(merged_attention)

    # small_feature_conv = tf.keras.layers.Conv2D(filters, kernel_size=(2, 2), padding="same", activation="relu")(input_tensor)   
    # small_feature_conv = tf.keras.layers.LayerNormalization()(small_feature_conv)


    out = tf.keras.layers.Add()([mlp_output, input_tensor])
    # out = tf.keras.layers.LayerNormalization()(out)




    x1 = tf.keras.layers.Dense(filters, activation="relu")(out)
    x1 = tf.keras.layers.Dense(filters)(x1)
    out_tensor = tf.keras.layers.Add()([out, x1])




        # #空间注意力
    # attn_output = SpatialAttentionLayer()(x)
    # x = tf.keras.layers.Add()([x, attn_output])

    # # 通道注意力
    # channel_attention = ChannelAttentionLayer()(x)
    # x = tf.keras.layers.Multiply()([x, channel_attention])

    # # MLP层
    # mlp_output = MLPBlock(units=filters)(x)
    # out = tf.keras.layers.Add()([mlp_output, input_tensor])


    #     # 空间注意力
    #     spatial_attention = SpatialAttentionLayer()(x)

    #     # 通道注意力
    #     channel_attention = ChannelAttentionLayer()(x)

    #     # 元素级相加合并注意力输出
    #     merged_attention = tf.keras.layers.Add()([spatial_attention, channel_attention,x])
    #    # x = tf.keras.layers.BatchNormalization()(merged_attention)  # 添加批次归一化
    #     # MLP层
    #     mlp_output = MLPBlock(units=filters)(merged_attention)
    #     out = tf.keras.layers.Add()([mlp_output, input_tensor])


    # mlp_output = MLPBlock(units=filters)(x)
    # # 空间注意力
    # spatial_attention = SpatialAttentionLayer()(mlp_output)

    # # 通道注意力
    # channel_attention = ChannelAttentionLayer()(mlp_output)
    # merged_attention = tf.keras.layers.Add()([spatial_attention, channel_attention])
    # merged_attention= tf.keras.layers.BatchNormalization()(merged_attention)  # 添加批次归一化
    # merged_attention=tf.keras.layers.Add()([merged_attention,mlp_output])
    # out = tf.keras.layers.Add()([merged_attention, x])

    # small_feature_conv = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same", activation="relu")(out)

    # out = tf.keras.layers.Add()([small_feature_conv, out])
    # # x1 = tf.keras.layers.Dense(filters, activation="swish")(out)







    return out_tensor



# def boundary_aggregation_module(input_tensor, filters):
#     # 使用Sobel滤波器来检测边界
#     sobel_x = tf.image.sobel_edges(input_tensor)[:,:,:,0]
#     sobel_y = tf.image.sobel_edges(input_tensor)[:,:,:,1]
#     sobel_output = tf.math.abs(sobel_x) + tf.math.abs(sobel_y)

#     # 使用Conv层进一步提取边界特征
#     edge_features = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), padding='same', activation='relu')(sobel_output)
    
#     # 使用一个1x1的Conv层来调整通道数量并与原始输入合并
#     edge_features = tf.keras.layers.Conv2D(filters, kernel_size=(1,1), padding='same', activation='relu')(edge_features)

#     # 合并原始特征和边界特征
#     merged = tf.keras.layers.Add()([input_tensor, edge_features])
    
#     return merged

def boundary_aggregation_module(input_tensor, filters):
    # 使用Sobel滤波器来检测边界
    sobel_x = tf.image.sobel_edges(input_tensor)[:,:,:,0]
    sobel_y = tf.image.sobel_edges(input_tensor)[:,:,:,1]
    sobel_output = tf.math.abs(sobel_x) + tf.math.abs(sobel_y)

    # 使用Conv层进一步提取边界特征
    edge_features = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), padding='same', activation='relu')(sobel_output)
    edge_features = tf.keras.layers.BatchNormalization()(edge_features)
    edge_features = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), padding='same', activation='relu')(edge_features)
    
    # 添加残差连接
    edge_residual = tf.keras.layers.Conv2D(filters, kernel_size=(1,1), padding='same', activation='relu')(sobel_output)
    edge_features = tf.keras.layers.Add()([edge_features, edge_residual])
    
    # 使用一个1x1的Conv层来调整通道数量
    edge_features = tf.keras.layers.Conv2D(filters, kernel_size=(1,1), padding='same', activation='relu')(edge_features)

    # 合并原始特征和边界特征
    merged = tf.keras.layers.Add()([input_tensor, edge_features])
    
    return merged



    # 边界预测网络
def boundary_prediction_subnetwork(x, filters):
    # 一系列卷积层
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters//2, (3, 3), padding="same", activation="relu")(x)
    # 输出边界概率图
    boundary_map = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)
    return boundary_map

class MLPBlock(tf.keras.layers.Layer):
    # def __init__(self, units, activation="swish",name=None,trainable=True,dtype=None):
    def __init__(self, units, activation="relu",name=None,trainable=True,dtype=None):
        super(MLPBlock, self).__init__(trainable=trainable,name=name,dtype=dtype)
        self.units = units
        self.activation = activation
        self.dense1 = tf.keras.layers.Dense(units, activation=activation)
        self.dense2 = tf.keras.layers.Dense(units, activation=None)  # No activation on the last layer
        # self.dense_layer1 = tf.keras.layers.Dense(self.units, 
        #                                           activation=self.activation, 
        #                                           kernel_regularizer=l2(0.01))
        
        # self.dense_layer2 = tf.keras.layers.Dense(self.units, 
        #                                           kernel_regularizer=l2(0.01))
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        # x = self.dense_layer1(inputs)
        # x = self.dense_layer2(x)  
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': self.activation
        })
        return config

class SpatialAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size=2, **kwargs):
        super(SpatialAttentionLayer, self).__init__()
        self.kernel_size = kernel_size
        # self.conv1 = tf.keras.layers.Conv2D(1, self.kernel_size, padding='same', activation='swish')
        # self.conv1 = tf.keras.layers.Conv2D(1, self.kernel_size, padding='same', activation='sigmoid')
        self.conv1 = tf.keras.layers.Conv2D(1, self.kernel_size, padding='same', activation='relu')
        
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

class ChannelAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, ratio=64, **kwargs):
        super(ChannelAttentionLayer, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        # self.shared_layer_one = tf.keras.layers.Dense(input_shape[-1] // self.ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        self.shared_layer_one = tf.keras.layers.Dense(input_shape[-1] // self.ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        self.shared_layer_two = tf.keras.layers.Dense(input_shape[-1], kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        super(ChannelAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        avg_pool = tf.keras.layers.Reshape((1, 1, avg_pool.shape[1]))(avg_pool)
        assert avg_pool.shape[1:] == (1, 1, inputs.shape[-1])
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
        max_pool = tf.keras.layers.Reshape((1, 1, max_pool.shape[1]))(max_pool)
        assert max_pool.shape[1:] == (1, 1, inputs.shape[-1])
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        cbam_feature = tf.keras.layers.Add()([avg_pool, max_pool])
        # cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)
        cbam_feature = tf.keras.layers.Activation('relu')(cbam_feature)

        return inputs * cbam_feature


    def get_config(self):
        config = super().get_config()
        config.update({
            "ratio": self.ratio,
        })
        return config

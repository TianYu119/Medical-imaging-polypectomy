import tensorflow as tf
from keras_cv_attention_models import caformer
from layers.upsampling import decode
from layers.convformer import convformer as FBEM
from layers.util_layers import merge, conv_bn_act
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
import tensorflow.keras.backend as K
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Multiply, GlobalAveragePooling2D, Reshape, Dense, Add
from layers.AttentionBoundaryFusionBlock import AttentionBoundaryFusionBlock
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout



def build_model(img_size = 256, num_classes = 1):
    backbone = caformer.CAFormerS18(input_shape=(256, 256, 3), pretrained="imagenet", num_classes = 0)
    
    layer_names = ['stack4_block3_mlp_Dense_1', 'stack3_block9_mlp_Dense_1', 'stack2_block3_mlp_Dense_1', 'stack1_block3_mlp_Dense_1']
    layers = [backbone.get_layer(x).output for x in layer_names]

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    x = layers[0]

        # 插入空间金字塔池化层
    spp_output = spatial_pyramid_pooling(x, [1, 2, 4])  # 可以选择不同的池化尺度
    x = tf.keras.layers.Concatenate(axis=channel_axis)([x, spp_output])



    upscale_feature = decode(x, scale = 4, filters = x.shape[channel_axis])

    for i, layer in enumerate(layers[1:]):
    # for i, layer in enumerate(layers[1:]):
        
        x = decode(x, scale = 2, filters = layer.shape[channel_axis])
        
        # layer_fusion = FBEM(layer, layer.shape[channel_axis])
        layer_fusion = FBEM(layer, layer.shape[channel_axis])
        # attention_boundary_output = AttentionBoundaryFusionBlock(layer, layer.shape[channel_axis])
        # layer_fusion = tf.keras.layers.Add()([layer_fusion_output, attention_boundary_output])

        
        ## Doing multi-level concatenation
        if (i%2 == 1):
            upscale_feature = tf.keras.layers.Conv2D(layer.shape[channel_axis], (1, 1), activation = "relu", padding = "same")(upscale_feature)
            x = tf.keras.layers.Add()([x, upscale_feature])
            x = tf.keras.layers.Conv2D(x.shape[channel_axis], (1, 1), activation = "relu", padding = "same")(x)
        
        # x = adaptive_interaction_module(x)
        x = merge([x, layer_fusion], layer.shape[channel_axis])
        x = conv_bn_act(x, layer.shape[channel_axis], (1, 1))

        ## Upscale for next level feature
        if (i%2 == 1):
            upscale_feature = decode(x, scale = 8, filters = layer.shape[channel_axis])
        
    filters = x.shape[channel_axis] //2
    upscale_feature = conv_bn_act(upscale_feature, filters, 1)
    x = adaptive_interaction_module(x)
    x = decode(x, filters, 4)
    x = tf.keras.layers.Add()([x, upscale_feature])
    x = conv_bn_act(x, filters, 1)
    x = Conv2D(num_classes, kernel_size=1, padding='same', activation='sigmoid')(x)
    model = Model(backbone.input, x)

    return model


def spatial_pyramid_pooling(inputs, pool_list):
    output = []
    for pool_size in pool_list:
        x = tf.keras.layers.AveragePooling2D(pool_size=pool_size)(inputs)
        x = tf.keras.layers.Conv2D(512, 1, padding='same')(x)
        x = tf.keras.layers.UpSampling2D(size=pool_size, interpolation='bilinear')(x)
        output.append(x)
    return tf.keras.layers.Concatenate(axis=-1)(output)


def se_block(input_tensor, ratio=16):
    """Squeeze and Excitation block."""
    filters = input_tensor.shape[-1]
    se_shape = (1, 1, filters)
    
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)

    return Multiply()([input_tensor, se])

def adaptive_interaction_module(inputs, dropout_rate=0.1):
    """Adaptive Interaction Module (AIM)."""
    # inputs = tf.keras.layers.LayerNormalization()(inputs)
    # Depthwise Separable Convolution
    x = DepthwiseConv2D(3, padding='same')(inputs)
    x = Conv2D(inputs.shape[-1], 1, padding='same', activation='relu')(x)
    
    # Dilated Convolution with dilation rate 2
    dilated_x = Conv2D(inputs.shape[-1], 3, dilation_rate=2, padding='same', activation='relu')(inputs)
    
    # Squeeze and Excitation block
    se_x = se_block(Add()([x, dilated_x]))
    
    # se_x = Dropout(dropout_rate)(se_x)

    return se_x


# def swish(x):
#     """Swish activation function"""
#     return x * Activation('sigmoid')(x)


# def adaptive_interaction_module(inputs, dropout_rate=0.2):
#     """Adaptive Interaction Module (AIM)."""
    
#     # Depthwise Separable Convolution
#     x = DepthwiseConv2D(3, padding='same')(inputs)
#     # x = BatchNormalization()(x)
#     # x = Activation(swish)(x)
#     x = Activation('relu')(x)
    
#     x = Conv2D(inputs.shape[-1], 1, padding='same')(x)
#     # x = BatchNormalization()(x)
#     # x = Activation(swish)(x)
#     x = Activation('relu')(x)
#     # Dilated Convolution with dilation rate 2
#     dilated_x = Conv2D(inputs.shape[-1], 3, dilation_rate=2, padding='same')(inputs)
#     # dilated_x = BatchNormalization()(dilated_x)
#     # dilated_x = Activation(swish)(dilated_x)
#     dilated_x = Activation('relu')(dilated_x)
#     # Merge and apply SE block
#     merged_x = Add()([x, dilated_x])
#     se_x = se_block(merged_x)
    
#     # Optional: Dropout for regularization
#     se_x = Dropout(dropout_rate)(se_x)
    
#     return se_x

import tensorflow as tf 
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
import numpy as np
def dice_coeff(y_true, y_pred):
    
    _epsilon = 10 ** -7
    intersections = tf.reduce_sum(y_true * y_pred)
    unions = tf.reduce_sum(y_true + y_pred)
    dice_scores = (2.0 * intersections + _epsilon) / (unions + _epsilon)
    
    return dice_scores

def dice_loss(y_true, y_pred):
    
    loss = 1 - dice_coeff(y_true, y_pred)
    
    return loss

def total_loss(y_true, y_pred):
    return 0.5*binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def IoU(y_true, y_pred, eps=1e-6):
    
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    
    return K.mean( (intersection + eps) / (union + eps), axis=0)

def zero_IoU(y_true, y_pred):
    
    return IoU(1-y_true, 1-y_pred)

def bce_dice_loss(y_true, y_pred):
    
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    
    y_true_pos = tf.reshape(y_true,[-1])
    y_pred_pos = tf.reshape(y_pred,[-1])
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    
    return 1 - tversky(y_true, y_pred)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    
    tv = tversky(y_true, y_pred)
    
    return K.pow((1 - tv), gamma)







def e_measure(y_true, y_pred):
    mu_x = tf.reduce_mean(y_pred)
    mu_y = tf.reduce_mean(y_true)
    sigma_x = tf.math.reduce_std(y_pred)
    sigma_y = tf.math.reduce_std(y_true)
    sigma_xy = tf.reduce_mean((y_pred - mu_x) * (y_true - mu_y))

    K1 = 0.01
    K2 = 0.03
    L = 1  # L is the dynamic range of the pixel-values (typically this is 2^number_of_bits_per_pixel - 1)
    C1 = (K1*L)**2
    C2 = (K2*L)**2

    ssim = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2))
    return ssim

def mean_e_measure(y_true, y_pred):
    return tf.reduce_mean(e_measure(y_true, y_pred))

def max_e_measure(y_true, y_pred):
    return tf.reduce_max(e_measure(y_true, y_pred))



def gradient(input_tensor):
    kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    kernel_x = tf.cast(tf.expand_dims(tf.expand_dims(kernel_x, axis=-1), axis=-1), tf.float32)
    kernel_y = tf.cast(tf.expand_dims(tf.expand_dims(kernel_y, axis=-1), axis=-1), tf.float32)
    
    gradient_x = tf.nn.conv2d(input_tensor, kernel_x, strides=[1, 1, 1, 1], padding="SAME")
    gradient_y = tf.nn.conv2d(input_tensor, kernel_y, strides=[1, 1, 1, 1], padding="SAME")
    
    gradient_mag = tf.sqrt(tf.square(gradient_x) + tf.square(gradient_y))
    
    return gradient_mag



def s_measure(y_true, y_pred, alpha=0.5):
    y_true_grad = gradient(y_true)
    y_pred_grad = gradient(y_pred)
    
    y_true_grad_norm = tf.nn.l2_normalize(y_true_grad, axis=[1, 2])
    y_pred_grad_norm = tf.nn.l2_normalize(y_pred_grad, axis=[1, 2])
    
    grad_dot = tf.reduce_sum(y_true_grad_norm * y_pred_grad_norm, axis=[1, 2])
    
    y_true_mean = tf.reduce_mean(y_true, axis=[1, 2])
    y_pred_mean = tf.reduce_mean(y_pred, axis=[1, 2])
    
    region = y_true_mean * y_pred_mean + (1 - y_true_mean) * (1 - y_pred_mean)
    
    s = alpha * grad_dot + (1 - alpha) * region
    
    return s



def weighted_f_measure(y_true, y_pred, beta=1.0):
    # Precision and Recall calculations
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1-y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1-y_pred))
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    
    # F-measure calculation
    f_measure = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-7)
    
    return f_measure
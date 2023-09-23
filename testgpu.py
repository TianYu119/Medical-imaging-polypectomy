import tensorflow as tf


if tf.test.gpu_device_name():
    print('GPU设备可用：{}'.format(tf.test.gpu_device_name()))
else:
    print('GPU设备不可用')

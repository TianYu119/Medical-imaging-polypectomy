from sklearn.model_selection import train_test_split
import tensorflow as tf  
from metrics.segmentation_metrics import dice_coeff, bce_dice_loss, zero_IoU, dice_loss, total_loss
from tensorflow.keras.utils import get_custom_objects
from metrics.segmentation_metrics import IoU as IoU
import os 
from callbacks.callbacks import get_callbacks, cosine_annealing_with_warmup
from dataloader.dataloader import build_augmenter, build_dataset, build_decoder
# from supervision.dataloader import build_augmenter, build_dataset, build_decoder
from model import build_model
import matplotlib.pyplot as plt
import tensorflow_addons as tfa  
from optimizers.lion_opt import Lion
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np


from kerastuner.tuners import RandomSearch

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

img_size = 256
BATCH_SIZE = 6
SEED = 42
save_path = "best_model.h5"

valid_size = 0.1
test_size = 0.15
epochs =400
save_weights_only = True
max_lr = 1e-4
min_lr = 1e-6


route = './TrainDataset'
X_path = './TrainDataset/image/'
Y_path = './TrainDataset/masks/'

# route = './Kvasir-SEG'
# X_path = './Kvasir-SEG/images/'
# Y_path = './Kvasir-SEG/masks/'

X_full = sorted(os.listdir(f'{route}/image'))
Y_full = sorted(os.listdir(f'{route}/masks'))

print(len(X_full))

# valid_size = 0.1
# test_size = 0.

X_train, X_valid = train_test_split(X_full, test_size=valid_size, random_state=SEED)
Y_train, Y_valid = train_test_split(Y_full, test_size=valid_size, random_state=SEED)

X_train, X_test = train_test_split(X_train, test_size=test_size, random_state=SEED)
Y_train, Y_test = train_test_split(Y_train, test_size=test_size, random_state=SEED)

X_train = [X_path + x for x in X_train]
X_valid = [X_path + x for x in X_valid]
X_test = [X_path + x for x in X_test]

Y_train = [Y_path + x for x in Y_train]
Y_valid = [Y_path + x for x in Y_valid]
Y_test = [Y_path + x for x in Y_test]

print("N Train:", len(X_train))
print("N Valid:", len(X_valid))
print("N test:", len(X_test))
# print(X_train)
train_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
train_dataset = build_dataset(X_train, Y_train, bsize=BATCH_SIZE, decode_fn=train_decoder, 
                            augmentAdv=False, augment=False, augmentAdvSeg=True)

valid_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
valid_dataset = build_dataset(X_valid, Y_valid, bsize=BATCH_SIZE, decode_fn=valid_decoder, 
                            augmentAdv=False, augment=False, repeat=False, shuffle=False,
                            augmentAdvSeg=False)

test_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
test_dataset = build_dataset(X_test, Y_test, bsize=BATCH_SIZE, decode_fn=test_decoder, 
                            augmentAdv=False, augment=False, repeat=False, shuffle=False,
                            augmentAdvSeg=False)


# 步骤 1: 导入Keras Tuner













# 步骤 2: 定义超参数搜索空间
def build_model_tun(hp):
    model = build_model(img_size)
    
    # 添加 weight_decay 到搜索空间
    weight_decay = hp.Choice('weight_decay', [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    
    # 使用带有 weight_decay 的 AdamW 优化器
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    
    model.compile(optimizer=optimizer, loss=dice_loss)
    return model


# 步骤 3: 创建评估函数
# 评估函数不需要改变，因为 weight_decay 是在模型编译时设置的

def evaluate_model(hp):
    model = build_model_tun(hp)
    history = model.fit(train_dataset, epochs=10, batch_size=6, validation_data=valid_dataset)
    
    val_preds = model.predict(valid_dataset)

    val_labels = []
    for _, labels in valid_dataset:
        val_labels.append(labels.numpy())
    val_labels = np.concatenate(val_labels, axis=0)

    custom_iou = IoU(val_labels, val_preds)
    zero_iou = zero_IoU(val_labels, val_preds)
    print("Custom IoU:", custom_iou)
    print("Zero IoU:", zero_iou)
    
    val_loss = history.history['val_loss'][-1]
    return val_loss


# 步骤 4: 执行超参数搜索
tuner = RandomSearch(
    build_model_tun,
    objective='val_loss',
    max_trials=10,
    directory='tuner_results',
    project_name='my_tuner')

tuner.search(
    x=train_dataset,
    epochs=10,
    validation_data=valid_dataset,
    verbose=1)












# # 步骤 2: 定义超参数搜索空间
# def build_model_tun(hp):
#     model = build_model(img_size)
#     learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
#     # 只传递损失函数
#     model.compile(optimizer=optimizer, loss=dice_loss)
#     return model


# # 步骤 3: 创建评估函数
# def evaluate_model(hp):
#     model = build_model_tun(hp)  
#     history = model.fit(train_dataset, epochs=10, batch_size=6, validation_data=valid_dataset)
    
#     # 在训练后，单独计算和记录自定义度量
#     val_preds = model.predict(valid_dataset)

#     val_labels = []
#     for _, labels in valid_dataset:
#         val_labels.append(labels.numpy())

# # 合并所有批次的标签
#     val_labels = np.concatenate(val_labels, axis=0)

#     custom_iou = IoU(val_labels, val_preds)
#     zero_iou = zero_IoU(val_labels, val_preds)
#     print("Custom IoU:", custom_iou)
#     print("Zero IoU:", zero_iou)
    
#     val_loss = history.history['val_loss'][-1]
#     return val_loss


# # 步骤 4: 执行超参数搜索
# tuner = RandomSearch(
#     build_model_tun,
#     objective='val_loss',
#     max_trials=10,
#     directory='tuner_results',
#     project_name='my_tuner')

# tuner.search(
#     x=train_dataset,
#     epochs=10,
#     validation_data=valid_dataset,
#     verbose=1)













from sklearn.model_selection import train_test_split
import tensorflow as tf  
from metrics.segmentation_metrics import dice_coeff, bce_dice_loss, IoU, zero_IoU, dice_loss, total_loss,s_measure, weighted_f_measure, mean_e_measure, max_e_measure
from tensorflow.keras.utils import get_custom_objects
import os 
from callbacks.callbacks import get_callbacks, cosine_annealing_with_warmup
from dataloader.dataloader import build_augmenter, build_dataset, build_decoder
# from supervision.dataloader import build_augmenter, build_dataset, build_decoder
from model import build_model
import matplotlib.pyplot as plt
import tensorflow_addons as tfa  
from optimizers.lion_opt import Lion
from tensorflow.keras.callbacks import ModelCheckpoint
import shutil
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"




img_size = 256
BATCH_SIZE = 4
SEED = 42
save_path = "best_model.h5"

valid_size = 0.1
test_size = 0.15
epochs =400
save_weights_only = True
max_lr = 1e-4
min_lr = 1e-6

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(cosine_annealing_with_warmup, verbose=0)
# opts = tfa.optimizers.AdamW(lr= 1e-3, weight_decay = lr_schedule)
# opts = tf.keras.optimizers.SGD(lr=1e-4)
# route = "./Kvasir-SEG/"
# X_path = '/root/tqhuy/Polyp/PEFNet-main/Kvasir-SEG/images/'
# Y_path = '/root/tqhuy/Polyp/PEFNet-main/Kvasir-SEG/masks/'

model = build_model(img_size)
def myprint(s):
    with open('modelsummary.txt','a') as f:
        print(s, file=f)

model.summary(print_fn=myprint)
model.summary()
# model = create_segment_model()
starter_learning_rate = 1e-4
end_learning_rate = 1e-6
decay_steps = 1000
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.2)

opts = tfa.optimizers.AdamW(learning_rate = 1e-4, weight_decay = learning_rate_fn)

get_custom_objects().update({"dice": dice_loss})
model.compile(optimizer = opts,
            loss='dice',
            metrics=[dice_coeff,bce_dice_loss, IoU, zero_IoU, weighted_f_measure,s_measure, mean_e_measure, max_e_measure,MeanAbsoluteError(name='mae')])

# model.summary()
route = './TrainDataset'
X_path = './TrainDataset/image/'
Y_path = './TrainDataset/masks/'

# route = './TestDataset//KV+CliDB/train'
# X_path = './TestDataset//KV+CliDB/train/images/'
# Y_path = './TestDataset//KV+CliDB/train/masks/'

X_full = sorted(os.listdir(f'{route}/image'))
Y_full = sorted(os.listdir(f'{route}/masks'))

print(len(X_full))

# valid_size = 0.1
# test_size = 0.

X_train, X_valid = train_test_split(X_full, test_size=valid_size, random_state=SEED)
Y_train, Y_valid = train_test_split(Y_full, test_size=valid_size, random_state=SEED)

X_train = [X_path + x for x in X_train]
X_valid = [X_path + x for x in X_valid]

Y_train = [Y_path + x for x in Y_train]
Y_valid = [Y_path + x for x in Y_valid]


print("N Train:", len(X_train))
print("N Valid:", len(X_valid))

# print(X_train)
train_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
train_dataset = build_dataset(X_train, Y_train, bsize=BATCH_SIZE, decode_fn=train_decoder, 
                            augmentAdv=False, augment=False, augmentAdvSeg=True)

valid_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
valid_dataset = build_dataset(X_valid, Y_valid, bsize=BATCH_SIZE, decode_fn=valid_decoder, 
                            augmentAdv=False, augment=False, repeat=False, shuffle=False,
                            augmentAdvSeg=False)


callbacks = get_callbacks(monitor = 'val_loss', mode = 'min', save_path = save_path, _max_lr = max_lr
                        , _min_lr = min_lr , _cos_anne_ep = 1000, save_weights_only = save_weights_only)

checkpoint = ModelCheckpoint("better_model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks.append(checkpoint)


# early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

# # 3. 将EarlyStopping对象添加到回调列表中
# callbacks.append(early_stopping)


steps_per_epoch = len(X_train) // BATCH_SIZE

print("START TRAINING:")


print(train_dataset)
his = model.fit(train_dataset, 
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_dataset)

# train_dataset = build_dataset(X_train, Y_train, bsize=BATCH_SIZE, decode_fn=train_decoder, 
#                             augmentAdv=False, augment=False, augmentAdvSeg=True)
# train_dataset = train_dataset.repeat()

# his = model.fit(train_dataset, 
#             epochs=epochs-150,
#             verbose=1,
#             callbacks=callbacks,
#             steps_per_epoch=steps_per_epoch,
#             validation_data=valid_dataset)
print(his)
model.load_weights(save_path)


# # 定义目标文件夹
# images_output_folder = './TrainDataset/Kvasir-SEG/test/images/'
# masks_output_folder = './TrainDataset/Kvasir-SEG/test/masks/'

# # 如果目标文件夹不存在，则创建
# if not os.path.exists(images_output_folder):
#     os.makedirs(images_output_folder)
    
# if not os.path.exists(masks_output_folder):
#     os.makedirs(masks_output_folder)

# # 遍历测试集中的每个图像和掩码，并将它们复制到目标文件夹
# for img_path, mask_path in zip(X_test, Y_test):
#     # 构建目标文件路径
#     img_dest_path = os.path.join(images_output_folder, os.path.basename(img_path))
#     mask_dest_path = os.path.join(masks_output_folder, os.path.basename(mask_path))
    
#     # 复制文件
#     shutil.copy(img_path, img_dest_path)
#     shutil.copy(mask_path, mask_dest_path)

# 从历史记录中提取训练损失、验证损失和准确率
train_loss = his.history['loss']
val_loss = his.history['val_loss']
train_accuracy = his.history['dice_coeff']
val_accuracy = his.history['val_dice_coeff']

# 创建两个子图，一个用于损失，一个用于准确率
plt.figure(figsize=(12, 6))

# 损失图表
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Time')

# 准确率图表
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Dice Coefficient')
plt.plot(val_accuracy, label='Validation Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend()
plt.title('Dice Coefficient Over Time')

# # 展示图表
# plt.show()

# 保存图表到文件
plt.savefig('training_plots.png')

# 关闭图表
plt.close()





# print(test_dataset)
# model.evaluate(test_dataset)

model.save("final_model.h5")






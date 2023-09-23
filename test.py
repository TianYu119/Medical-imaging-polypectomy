from sklearn.model_selection import train_test_split
import tensorflow as tf  
from metrics.segmentation_metrics import dice_coeff, bce_dice_loss, IoU, zero_IoU, dice_loss, total_loss
from tensorflow.keras.utils import get_custom_objects
import os 
from callbacks.callbacks import get_callbacks, cosine_annealing_with_warmup
from dataloader.dataloader import build_augmenter, build_dataset, build_decoder
# from supervision.dataloader import build_augmenter, build_dataset, build_decoder
from model import build_model
import tensorflow_addons as tfa  
from optimizers.lion_opt import Lion
import cv2
import numpy as np
from sklearn.metrics import jaccard_score, mean_absolute_error
from layers.convformer import MLPBlock



os.environ["CUDA_VISIBLE_DEVICES"]="2"
img_size = 256
BATCH_SIZE = 8
SEED = 42
save_path = "best_model.h5"

valid_size = 0.1
test_size = 0.15
epochs = 350
save_weights_only = True
max_lr = 1e-4
min_lr = 1e-6

# route = './TestDataset/CVC-300'
# X_path = './TestDataset/CVC-300/images/'
# Y_path = './TestDataset/CVC-300/masks/'
# X_full = sorted(os.listdir(f'{route}/images'))
# Y_full = sorted(os.listdir(f'{route}/masks'))

# route = './TestDataset/ETIS-LaribPolypDB'
# X_path = './TestDataset/ETIS-LaribPolypDB/images/'
# Y_path = './TestDataset/ETIS-LaribPolypDB/masks/'
# X_full = sorted(os.listdir(f'{route}/images'))
# Y_full = sorted(os.listdir(f'{route}/masks'))


# route = './TestDataset/CVC-ColonDB'
# X_path = './TestDataset/CVC-ColonDB/images/'
# Y_path = './TestDataset/CVC-ColonDB/masks/'
# X_full = sorted(os.listdir(f'{route}/images'))
# Y_full = sorted(os.listdir(f'{route}/masks'))


route = './TestDataset/CVC-ClinicDB'
X_path = './TestDataset/CVC-ClinicDB/images/'
Y_path = './TestDataset/CVC-ClinicDB/masks/'
X_full = sorted(os.listdir(f'{route}/images'))
Y_full = sorted(os.listdir(f'{route}/masks'))


# X_train, X_valid = train_test_split(X_full, test_size=valid_size, random_state=SEED)
# Y_train, Y_valid = train_test_split(Y_full, test_size=valid_size, random_state=SEED)

# X_train, X_test = train_test_split(X_train, test_size=test_size, random_state=SEED)
# Y_train, Y_test = train_test_split(Y_train, test_size=test_size, random_state=SEED)

X_test = train_test_split(X_full, test_size=valid_size, random_state=SEED, shuffle=False)
Y_test= train_test_split(Y_full, test_size=valid_size, random_state=SEED, shuffle=False)

print("=========================================")
print(X_test)
print("=========================================")
print(Y_test)
print("=========================================")



# 将嵌套的列表扁平化成一个单层的列表
X_test = [x for sublist in X_test for x in sublist]
Y_test = [y for sublist in Y_test for y in sublist]

# 然后将路径添加到每个文件名
X_test = [X_path + '/' + x for x in X_test]
Y_test = [Y_path + '/' + y for y in Y_test]





# # X_test = ['./TestDataset/CVC-300/images/149.png']
# test_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='png', segment=True, ext2='png')
# test_dataset = build_dataset(X_test, Y_test, bsize=BATCH_SIZE, decode_fn=test_decoder, 
#                             augmentAdv=False, augment=False, repeat=False, shuffle=False,
#                             augmentAdvSeg=False)


# model_path = 'best_model.h5'  # 模型文件的路径
# loaded_model = tf.keras.models.load_model(model_path)

# # 准备输入数据（示例）


# # 使用加载的模型进行预测
# predictions = loaded_model.predict(test_dataset)

# # 处理预测结果
# print(predictions)


# 加载模型
model_path = 'better_model.h5'  # 模型文件的路径
custom_objects={"dice": dice_loss, "dice_coeff": dice_coeff, "bce_dice_loss": bce_dice_loss, "IoU": IoU,"zero_IoU":zero_IoU}
custom_objects["MLPBlock"] = MLPBlock 
loaded_model = tf.keras.models.load_model(model_path,custom_objects=custom_objects )

loaded_model.summary()
# 准备测试数据集
test_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
test_dataset = build_dataset(X_test, Y_test, bsize=BATCH_SIZE, decode_fn=test_decoder, 
                            augmentAdv=False, augment=False, repeat=False, shuffle=False,
                            augmentAdvSeg=False)

# 使用加载的模型进行预测
predictions = loaded_model.predict(test_dataset)

# 处理预测结果
print(X_test)


threshold = 0.3  # 你可以根据需要调整阈值

# 目标图像大小
target_height = 966
target_width = 1225

# 初始化一个高斯滤波器，可以调整卷积核的大小
gaussian_filter = cv2.getGaussianKernel(ksize=5, sigma=0)  # 调整ksize和sigma来控制平滑程度

# 定义形态学操作的核，用于去噪声
kernel = np.ones((5, 5), np.uint8)  # 调整核的大小来控制去噪声程度

# 遍历预测结果
segmented_images = []

for prediction, true_mask_path in zip(predictions, Y_test):
    # 将预测掩码转换为二进制图像
    binary_mask = (prediction > threshold).astype(np.uint8)
    
    # 对二进制图像进行高斯平滑
    smoothed_mask = cv2.filter2D(binary_mask, -1, gaussian_filter)

    # 可选：对平滑后的图像进行形态学开操作去噪声
    opened_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel)
    
    # 调整输出图像的大小，确保与目标大小一致
    resized_segmentation = cv2.resize(opened_mask * 255, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    
    
    segmented_images.append(resized_segmentation)

# 保存分割结果，使用 Y_test 中的文件名命名图像
for i, (segmented_image, true_mask_path) in enumerate(zip(segmented_images, Y_test)):
    file_name = os.path.basename(true_mask_path)  # 获取文件名
    output_path = os.path.join("output", f"segmentation_{file_name}")  # 构建保存路径
    cv2.imwrite(output_path, segmented_image)  # 保存图像




# 文件夹路径
output_folder = "output"
masks_folder = Y_path

# 获取output文件夹中的所有文件名并按名称排序
output_files = os.listdir(output_folder)
output_files.sort()

# 初始化IoU、Dice和MAE分数列表
iou_scores = []
dice_scores = []
mae_scores = []

# 遍历output文件夹中的每个文件
for output_file in output_files:
    # 构建输出图像和真实标签图像的完整文件路径
    output_path = os.path.join(output_folder, output_file)
    mask_name = output_file.replace("segmentation_", "")
    mask_path = os.path.join(masks_folder, mask_name)

    # 读取输出图像和真实标签图像
    output_image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    output_image = cv2.resize(output_image, (mask.shape[1], mask.shape[0]))

    # 二值化输出图像和真实标签图像（根据需要调整阈值）
    threshold = 128
    binary_output = (output_image > threshold).astype(np.uint8)
    binary_mask = (mask > threshold).astype(np.uint8)

    # 计算IoU
    intersection = np.logical_and(binary_mask, binary_output)
    union = np.logical_or(binary_mask, binary_output)
    iou = np.sum(intersection) / np.sum(union)
    iou_scores.append(iou)

    # 计算Dice系数
    dice = 2 * np.sum(intersection) / (np.sum(binary_mask) + np.sum(binary_output))
    dice_scores.append(dice)

    # 计算MAE
    mae = mean_absolute_error(binary_mask.flatten(), binary_output.flatten())
    mae_scores.append(mae)

    # 打印结果和文件名
    print(f"File: {output_file}")
    print(f"IoU: {iou}")
    print(f"Dice: {dice}")
    print(f"MAE: {mae}")
    print("-----------------")

# 计算平均值
mean_iou = np.mean(iou_scores)
mean_dice = np.mean(dice_scores)
mean_mae = np.mean(mae_scores)

# 打印平均结果
print(f"Mean IoU: {mean_iou}")
print(f"Mean Dice: {mean_dice}")
print(f"Mean MAE: {mean_mae}")



# # 文件夹路径
# output_folder = "output"
# masks_folder = "masks"

# # 初始化IoU、Dice和MAE分数列表
# iou_scores = []
# dice_scores = []
# mae_scores = []

# # 获取output文件夹中的所有文件名
# output_files = os.listdir(output_folder)

# # 遍历output文件夹中的每个文件
# for output_file in output_files:
#     # 构建输出图像和真实标签图像的完整文件路径
#     output_path = os.path.join(output_folder, output_file)
#     mask_path = os.path.join(masks_folder, output_file.replace("segmentation_", ""))  # 通过文件名匹配找到对应的mask文件

#     print(output_path)
#     print(mask_path)
#     # 读取输出图像和真实标签图像
#     output_image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#     # 二值化输出图像和真实标签图像（根据需要调整阈值）
#     threshold = 128
#     binary_output = (output_image > threshold).astype(np.uint8)
#     binary_mask = (mask > threshold).astype(np.uint8)

#     # 计算IoU
#     intersection = np.logical_and(binary_mask, binary_output)
#     union = np.logical_or(binary_mask, binary_output)
#     iou = np.sum(intersection) / np.sum(union)
#     iou_scores.append(iou)

#     # 计算Dice系数
#     dice = 2 * np.sum(intersection) / (np.sum(binary_mask) + np.sum(binary_output))
#     dice_scores.append(dice)

#     # 计算MAE
#     mae = mean_absolute_error(binary_mask.flatten(), binary_output.flatten())
#     mae_scores.append(mae)

# # 计算平均值
# mean_iou = np.mean(iou_scores)
# mean_dice = np.mean(dice_scores)
# mean_mae = np.mean(mae_scores)

# # 打印结果
# print(f"Mean IoU: {mean_iou}")
# print(f"Mean Dice: {mean_dice}")
# print(f"Mean MAE: {mean_mae}")


# # 初始化IoU、Dice和MAE分数列表
# iou_scores = []
# dice_scores = []
# mae_scores = []




# # 读取预测结果和真实标签
# prediction = cv2.imread("output/segmentation_186.png", cv2.IMREAD_GRAYSCALE)
# true_mask = cv2.imread("TestDataset/ETIS-LaribPolypDB/masks/186.png", cv2.IMREAD_GRAYSCALE)

# # 二值化预测结果（根据需要调整阈值）
# threshold = 128
# binary_prediction = (prediction > threshold).astype(np.uint8)
# binary_true_mask = (true_mask > threshold).astype(np.uint8)

# # 计算IoU
# intersection = np.logical_and(binary_true_mask, binary_prediction)
# union = np.logical_or(binary_true_mask, binary_prediction)
# iou = np.sum(intersection) / np.sum(union)

# # 计算Dice系数
# dice = 2 * np.sum(intersection) / (np.sum(binary_true_mask) + np.sum(binary_prediction))

# # 计算MAE
# mae = mean_absolute_error(binary_true_mask.flatten(), binary_prediction.flatten())

# # 打印结果
# print(f"IoU: {iou}")
# print(f"Dice: {dice}")
# print(f"MAE: {mae}")
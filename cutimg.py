import os
import shutil
import cv2
# def merge_folders(src_folder1, src_folder2, dst_folder):
#     os.makedirs(dst_folder, exist_ok=True)

#     for src_folder in [(src_folder1, 'Dataset1'), (src_folder2, 'Dataset2')]:  # 包装源文件夹和数据集名称
#         folder_path, dataset_name = src_folder
#         for root, dirs, files in os.walk(folder_path):
#             for file in files:
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#                     src_path = os.path.join(root, file)
#                     dst_path = os.path.join(dst_folder, f"{dataset_name}_{file}")
#                     counter = 1
#                     while os.path.exists(dst_path):
#                         name, ext = os.path.splitext(file)
#                         dst_path = os.path.join(dst_folder, f"{dataset_name}_{name}_{counter}{ext}")
#                         counter += 1

#                     # 假设标记图像的命名约定是在图像文件名后添加"_label"，并具有相同的文件扩展名
#                     label_src_path = os.path.join(root, f"{os.path.splitext(file)[0]}_label{os.path.splitext(file)[1]}")
#                     if os.path.exists(label_src_path):
#                         label_dst_path = os.path.join(dst_folder, f"{dataset_name}_{os.path.splitext(dst_path)[0]}_label{os.path.splitext(dst_path)[1]}")
#                         shutil.copy(label_src_path, label_dst_path)

#                     shutil.copy(src_path, dst_path)
# # 使用方法：
# src_folder1 = './TestDataset/ETIS-LaribPolypDB/images/'
# src_folder2 = './TestDataset/CVC-ColonDB/images/'
# dst_folder = './TestDataset/CODB+ET/images/'
# merge_folders(src_folder1, src_folder2, dst_folder)




import random


def split_dataset(dataset_folder, train_folder, test_folder, num_train, num_test):
    os.makedirs(os.path.join(train_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'masks'), exist_ok=True)

    images_folder = os.path.join(dataset_folder, 'images')
    masks_folder = os.path.join(dataset_folder, 'masks')

    images = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    random.shuffle(images)  # 随机打乱图像列表

    # 将指定数量的图像和掩膜移动到训练和测试文件夹
    for i, image in enumerate(images):
        src_image_path = os.path.join(images_folder, image)
        src_mask_path = os.path.join(masks_folder, image)  # 假设掩膜文件与图像文件具有相同的文件名

        if i < num_train:
            dst_image_path = os.path.join(train_folder, 'images', image)
            dst_mask_path = os.path.join(train_folder, 'masks', image)
        elif i < num_train + num_test:
            dst_image_path = os.path.join(test_folder, 'images', image)
            dst_mask_path = os.path.join(test_folder, 'masks', image)
        else:
            continue  # 如果已经移动了指定数量的图像和掩膜，就跳出循环
        
        shutil.copy(src_image_path, dst_image_path)  # 使用shutil.copy复制图像
        shutil.copy(src_mask_path, dst_mask_path)  # 使用shutil.copy复制掩膜

# 定义数据集文件夹、训练文件夹和测试文件夹的路径
dataset_folder_kvasir = './TestDataset/Kvasir-SEG'
train_folder_kvasir = './TestDataset/KV+CliDB/train'
test_folder_kvasir = './TestDataset/KV+CliDB/test'
dataset_folder_ClinicDB='./TestDataset/CVC-ClinicDB'
# 分割Kvasir-SEG数据集
split_dataset(dataset_folder_kvasir, train_folder_kvasir, test_folder_kvasir, 900, 100)
# 分割ClinicDB数据集
split_dataset(dataset_folder_ClinicDB,train_folder_kvasir,test_folder_kvasir, 548, 64)

# input_directory = './TestDataset/CVC-ClinicDB/term/'
# output_directory = './TestDataset/CVC-ClinicDB/masks/'
# # 遍历目录中的所有.tif文件
# for filename in os.listdir(input_directory):
#     if filename.endswith(".tif"):
#         # 读取.tif图像
#         image_path = os.path.join(input_directory, filename)
#         image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 使用IMREAD_UNCHANGED以确保读取.tif的所有通道

#         # 将.tif文件名转换为.jpg文件名
#         output_filename = filename.replace('.tif', '.png')
#         output_path = os.path.join(output_directory, output_filename)

#         # 保存为.jpg格式
#         cv2.imwrite(output_path, image)

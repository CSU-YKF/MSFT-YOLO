import os
import shutil
import random

# 设置随机种子以确保可重复性
random.seed(1)

# 数据集的根目录
root_dir = '../datasets/NEU-DET'

# 创建新的目录结构
sub_dirs = ['images', 'labels']
sub_sub_dirs = ['train', 'val', 'test']
for sub_dir in sub_dirs:
    for sub_sub_dir in sub_sub_dirs:
        os.makedirs(os.path.join(root_dir, sub_dir, sub_sub_dir), exist_ok=True)

# 获取所有图像和标签文件
image_files = sorted([f for f in os.listdir(os.path.join(root_dir, '../datasets/NEU-DET/images')) if f.endswith('.jpg')])
label_files = sorted([f for f in os.listdir(os.path.join(root_dir, '../datasets/NEU-DET/labels')) if f.endswith('.txt')])

# 确保图像和标签文件数量相同
assert len(image_files) == len(label_files)

# 打乱文件列表
combined = list(zip(image_files, label_files))
random.shuffle(combined)
image_files[:], label_files[:] = zip(*combined)

# 计算训练、验证和测试集的大小
total_files = len(image_files)
train_size = int(total_files * 0.7)
val_size = int(total_files * 0.2)
test_size = total_files - train_size - val_size

# 分配文件到新的目录结构
for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
    if i < train_size:
        destination = 'train'
    elif i < train_size + val_size:
        destination = 'val'
    else:
        destination = 'test'

    shutil.copy(os.path.join(root_dir, '../datasets/NEU-DET/images', image_file), os.path.join(root_dir,
                                                                                               '../datasets/NEU-DET/images', destination, image_file))
    shutil.copy(os.path.join(root_dir, '../datasets/NEU-DET/labels', label_file), os.path.join(root_dir,
                                                                                               '../datasets/NEU-DET/labels', destination, label_file))
import os
import shutil
import random

# 设置随机种子以确保可重复性
random.seed(1)

# 数据集的根目录
root_dir = '../datasets/NEU-DET'

# 创建新的目录结构
sub_dirs = ['images', 'labels']
sub_sub_dirs = ['train', 'val', 'test']
for sub_dir in sub_dirs:
    for sub_sub_dir in sub_sub_dirs:
        os.makedirs(os.path.join(root_dir, sub_dir, sub_sub_dir), exist_ok=True)

# 获取所有图像和标签文件
image_files = sorted([f for f in os.listdir(os.path.join(root_dir, '../datasets/NEU-DET/images')) if f.endswith('.jpg')])
label_files = sorted([f for f in os.listdir(os.path.join(root_dir, '../datasets/NEU-DET/labels')) if f.endswith('.txt')])

# 确保图像和标签文件数量相同
assert len(image_files) == len(label_files)

# 打乱文件列表
combined = list(zip(image_files, label_files))
random.shuffle(combined)
image_files[:], label_files[:] = zip(*combined)

# 计算训练、验证和测试集的大小
total_files = len(image_files)
train_size = int(total_files * 0.7)
val_size = int(total_files * 0.2)
test_size = total_files - train_size - val_size

# 分配文件到新的目录结构
for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
    if i < train_size:
        destination = 'train'
    elif i < train_size + val_size:
        destination = 'val'
    else:
        destination = 'test'

    shutil.copy(os.path.join(root_dir, '../datasets/NEU-DET/images', image_file), os.path.join(root_dir,
                                                                                               '../datasets/NEU-DET/images', destination, image_file))
    shutil.copy(os.path.join(root_dir, '../datasets/NEU-DET/labels', label_file), os.path.join(root_dir,
                                                                                               '../datasets/NEU-DET/labels', destination, label_file))

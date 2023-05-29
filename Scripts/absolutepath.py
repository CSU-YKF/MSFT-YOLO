import os

# 数据集的根目录
root_dir = '../datasets/NEU-DET'

# 创建新的目录结构
sub_dirs = ['images', 'labels']
sub_sub_dirs = ['train', 'val', 'test']

# 对于每个子目录（train、val、test），创建一个txt文件
for sub_sub_dir in sub_sub_dirs:
    with open(f'{root_dir}/{sub_sub_dir}.txt', 'w') as f:
        # 获取图像和标签文件的路径
        image_files = sorted(os.path.join(root_dir, '../datasets/NEU-DET/images', sub_sub_dir, f)
                             for f in os.listdir(os.path.join(root_dir, '../datasets/NEU-DET/images', sub_sub_dir)) if f.endswith('.jpg'))
        label_files = sorted(os.path.join(root_dir, '../datasets/NEU-DET/labels', sub_sub_dir, f)
                             for f in os.listdir(os.path.join(root_dir, '../datasets/NEU-DET/labels', sub_sub_dir)) if f.endswith('.txt'))

        # 确保图像和标签文件数量相同
        assert len(image_files) == len(label_files)

        # 将图像和标签文件的路径写入txt文件
        for image_file, label_file in zip(image_files, label_files):
            f.write(f'{image_file}\n')
            f.write(f'{label_file}\n')

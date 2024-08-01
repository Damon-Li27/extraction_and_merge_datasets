import h5py
import numpy as np


def merge_h5_files(file1, file2, output_file):
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        # 读取第一个文件的数据
        images1 = f1['images'][:]
        labels1 = f1['labels'][:]

        # 读取第二个文件的数据
        images2 = f2['train_set_x'][:]
        labels2 = f2['train_set_y'][:]

        # 合并数据
        combined_images = np.concatenate((images1, images2), axis=0)
        combined_labels = np.concatenate((labels1, labels2), axis=0)

    # 创建新的HDF5文件并写入合并后的数据
    with h5py.File(output_file, 'w') as f_out:
        f_out.create_dataset('train_set_x', data=combined_images)
        f_out.create_dataset('train_set_y', data=combined_labels)


file1 = 'datasets/dataset.h5'  # 替换为你的第一个文件路径
file2 = 'datasets/train_signs.h5'  # 替换为你的第二个文件路径
output_file = 'datasets/merged_dataset.h5'  # 替换为你希望输出的合并文件路径

merge_h5_files(file1, file2, output_file)
import os
import h5py
import numpy as np
from PIL import Image

def load_and_process_image(image_path, target_size=(64, 64)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image)
    # 如果是灰度图像，将其转换为 RGB 格式
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    return image_array / 255.0

def extract_label(image_name):
    # 标签在文件名的倒数第两个字符
    return int(image_name[-6:-5])

def create_h5_dataset(image_dir, h5_file_path, target_size=(64, 64)):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    num_images = len(image_files)

    # 创建HDF5文件
    with h5py.File(h5_file_path, 'w') as h5f:
        # 创建数据集
        images_dataset = h5f.create_dataset('images', (num_images, target_size[0], target_size[1], 3), dtype='float32')
        labels_dataset = h5f.create_dataset('labels', (num_images,), dtype='int')

        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_dir, image_file)
            image_array = load_and_process_image(image_path, target_size)
            label = extract_label(image_file)

            images_dataset[i] = image_array
            labels_dataset[i] = label

            if i % 100 == 0:
                print(f"Processed {i}/{num_images} images")

image_directory = 'train'  # 替换为你的图片目录
output_h5_file = 'datasets/dataset.h5'  # 替换为你希望输出的HDF5文件路径

create_h5_dataset(image_directory, output_h5_file)
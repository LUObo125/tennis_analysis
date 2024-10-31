# prompt: 把这些视频文件分成训练集和验证集，比例为3:1，同时生成对应的label

import os
import random
import shutil

def split_videos_and_create_labels(video_dir, annotation_file_path, train_dir, val_dir, train_label_path, val_label_path, train_ratio=0.75):
  """
  将视频文件分成训练集和验证集，并生成对应的标签文件。

  Args:
    video_dir: 视频文件目录。
    annotation_file_path: 视频注释文件路径。
    train_dir: 训练集目录。
    val_dir: 验证集目录。
    train_label_path: 训练集标签文件路径。
    val_label_path: 验证集标签文件路径。
    train_ratio: 训练集比例，默认为0.75。
  """

  # 读取注释文件
  with open(annotation_file_path, 'r') as f:
    lines = f.readlines()

  # 将视频文件和标签分成训练集和验证集
  video_data = []
  for line in lines:
    parts = line.strip().split()
    if len(parts) >= 2:
      video_file = parts[0]
      label = ' '.join(parts[1:])  # 标签可以是多个单词
      video_data.append((video_file, label))

  random.shuffle(video_data)
  num_train = int(len(video_data) * train_ratio)
  train_data = video_data[:num_train]
  val_data = video_data[num_train:]

  train_data.sort(key=lambda x: x[0])
  val_data.sort(key=lambda x: x[0])

  # 创建训练集和验证集目录
  os.makedirs(train_dir, exist_ok=True)
  os.makedirs(val_dir, exist_ok=True)

  # 复制视频文件到对应的目录并生成标签文件
  with open(train_label_path, 'w') as train_label_file, \
       open(val_label_path, 'w') as val_label_file:
    for video_file, label in train_data:
      src_path = os.path.join(video_dir, video_file)
      dst_path = os.path.join(train_dir, video_file)
      shutil.copy(src_path, dst_path)
      train_label_file.write(f'{video_file} {label}\n')

    for video_file, label in val_data:
      src_path = os.path.join(video_dir, video_file)
      dst_path = os.path.join(val_dir, video_file)
      shutil.copy(src_path, dst_path)
      val_label_file.write(f'{video_file} {label}\n')




if __name__ == "__main__":
    video_directory = 'output_video'  # 替换为您的视频目录
    annotation_file = 'output_video\label.csv'  # 替换为您的注释文件路径
    train_video_dir = 'train'  # 替换为您的训练集视频目录
    val_video_dir = 'val'  # 替换为您的验证集视频目录
    train_label_file = 'train/label_train.txt'  # 替换为您的训练集标签文件路径
    val_label_file = 'val/label_val.txt'  # 替换为您的验证集标签文件路径

    split_videos_and_create_labels(video_directory, annotation_file, train_video_dir, val_video_dir, train_label_file, val_label_file)

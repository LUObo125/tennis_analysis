# prompt: 我删除了一些视频文件，现在他们的视频标题不是连续的了，我想要将这些视频文件重新命名为从1开始的连续名称，并同步修改annotation文件

import os

def rename_videos_and_update_annotation(new_video_dir, video_dir, annotation_file_path):
  """
  重新命名视频文件并更新注释文件。

  Args:
    video_dir: 视频文件目录。
    annotation_file_path: 视频注释文件路径。
  """

  # 获取视频文件列表
  video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]  # 假设视频文件扩展名为.mp4

  # 重新命名视频文件并更新注释文件
  new_annotation_lines = []
  with open(annotation_file_path, 'r') as f:
    lines = f.readlines()
  for i, video_file in enumerate(video_files):
    new_video_name = f'{i+1}.mp4'
    old_video_path = os.path.join(video_dir, video_file)
    new_video_path = os.path.join(new_video_dir, new_video_name)
    os.rename(old_video_path, new_video_path)

    for line in lines:
      parts = line.strip().split()
      if len(parts) >= 2 and parts[0] == video_file:
        new_line = f'{new_video_name} {" ".join(parts[1:])}\n'
        new_annotation_lines.append(new_line)
        break


  with open(annotation_file_path, 'w') as f:
    f.writelines(new_annotation_lines)


# 使用示例



if __name__ == "__main__":
  video_directory = 'output'  # 替换为您的视频目录
  new_video_dir = 'output_video'
  annotation_file = 'output\label.csv'  # 替换为您的注释文件路径

  rename_videos_and_update_annotation(new_video_dir, video_directory, annotation_file) 
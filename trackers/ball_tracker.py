from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd

def calculate_distance(arr_list):
  result = []
  last_valid_arr = None

  for arr in arr_list:
    if arr:  # 如果数组非空
      if last_valid_arr is not None:
        dist1 = ((arr[0] - last_valid_arr[0]) ** 2 + (arr[1] - last_valid_arr[1]) ** 2) ** 0.5
        dist2 = ((arr[2] - last_valid_arr[2]) ** 2 + (arr[3] - last_valid_arr[3]) ** 2) ** 0.5
        result.append([dist1, dist2])
      else:
        result.append(arr)  # 第一个非空数组，直接添加到结果中
      last_valid_arr = arr
    else:
      result.append([])  # 空数组，添加到结果中

  return result

def calculate_distances(lst, threshold):
  """
  计算非空数组与前序最近非空数组中两点距离，并将超过阈值的数组置空。

  Args:
    lst: 包含多个数组的列表，每个数组表示两个点[x1, y1, x2, y2]。
    threshold: 距离阈值。

  Returns:
    处理后的列表，其中超过阈值的数组被置空。
  """

  copyoflst = lst.copy()
  result = copyoflst
  prev_non_empty = None
  i_pre = None

  #while True:
  for i in range(len(result)):
    if result[i]:  # 如果当前数组非空
        if prev_non_empty:
            # 计算距离
            dist1 = ((result[i][0] - prev_non_empty[0]) ** 2 + (result[i][1] - prev_non_empty[1]) ** 2) ** 0.5 / (i-i_pre)
            dist2 = ((result[i][2] - prev_non_empty[2]) ** 2 + (result[i][3] - prev_non_empty[3]) ** 2) ** 0.5 / (i-i_pre)

            # 如果距离超过阈值，将当前数组置空
            if dist1 > threshold or dist2 > threshold:
                result[i] = []

        prev_non_empty = result[i]  # 更新前序非空数组
        i_pre = i
    """     if result == copyoflst:  # 如果列表不再变化，说明所有距离都满足条件
        break
    copyoflst = result """
  return result

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]

        
        while True:
            distance = calculate_distance(ball_positions)
            new_lst = calculate_distances(ball_positions, 100)
            if new_lst == ball_positions:  # 如果列表不再变化，说明所有距离都满足条件
                break
            ball_positions = new_lst
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self,frame):
        results = self.model.predict(frame,conf=0.15, device=0)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    
import os
import time
from utils import (read_video, 
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters,
                   get_video_properties
                   )
import constants
from trackers import PlayerTracker,BallTracker,BallDetector
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
from copy import deepcopy

def get_video_properties(video):
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # get videos properties
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        length = int(video.get(cv2.cv.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.cv.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.cv.CAP_PROP_FRAME_HEIGHT))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, length, v_width, v_height

def cross_merge_lists(list1, list2):
    # 确定最长列表的长度
    max_length = max(len(list1), len(list2))
    merged_list = []
    #确保列表元素递增
    if list1[0]<list2[0]:
        first_stroke = 1
        for i in range(max_length):
            if i < len(list1):
                merged_list.append(list1[i])
            if i < len(list2):
                merged_list.append(list2[i])
    else:
        first_stroke = 2
        for i in range(max_length):
            if i < len(list2):
                merged_list.append(list2[i])
            if i < len(list1):
                merged_list.append(list1[i])
    merged_list.sort()
    return merged_list, first_stroke


def append_video_annotation(file_name, label, annotation_file_path):
  """
  将视频文件和标签添加到视频注释文件中。

  Args:
    file_name: 视频文件名。
    label: 视频标签。
    annotation_file_path: 视频注释文件路径。
  """
  with open(annotation_file_path, 'a') as f:
    f.write(f'{file_name} {label}\n')


def main():
    # 定义视频文件夹路径
    video_folder = 'F:/Gitstore/tennis-court-detector/output'
    output_folder = './output'
    annotation_file_path = './output/label.csv'

    if not os.path.exists(annotation_file_path):
        with open(annotation_file_path, 'w') as f:
            pass  # 创建空文件

    # 获取文件夹中所有视频文件
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    existing_files = [f for f in os.listdir('output') if f.endswith('.mp4')]
    #next_file_number = len(existing_files)+1

    # 循环处理每个视频文件
    for i, video_file in enumerate(video_files):
        print("processing video ", video_file)
        next_file_number = 0
        # Read Video
        input_video_path = os.path.join(video_folder, video_file)
        video = cv2.VideoCapture(input_video_path)
        fps, length, width, height = get_video_properties(video)
        video_frames = read_video(video)

        # Detect Players and Ball
        player_tracker = PlayerTracker(model_path='yolov8x')
        ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
        ball_detector = BallDetector('trackers/tracknet_weights_2_classes.pth', out_channels=2)

        # Court Line Detector model
        court_model_path = "models/keypoints_model.pth"
        court_line_detector = CourtLineDetector(court_model_path)
        court_keypoints = court_line_detector.predict(video_frames[0])

        player_detections = player_tracker.detect_frames(video_frames,
                                                        read_from_stub=False,
                                                        stub_path="tracker_stubs/player_detections.pkl"
                                                        )
        """ ball_detections = ball_tracker.detect_frames(video_frames,
                                                        read_from_stub=True,
                                                        stub_path="tracker_stubs/ball_detections.pkl"
                                                        )
        ball_detections = ball_tracker.interpolate_ball_positions(ball_detections, True) """
        
        ball_detections_detec = ball_detector.ball_detect_allframe(video_frames)
        ball_detections = ball_tracker.interpolate_ball_positions(ball_detections_detec, True)
        try:
        # choose players
            player_detection = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

            # MiniCourt
            mini_court = MiniCourt(video_frames[0]) 

            # Detect ball shots
            ball_shot_frames_p1, ball_shot_frames_p2 = ball_tracker.get_ball_shot_frames(ball_detections, player_detection)

        
            ball_shot_frames, first_stroke = cross_merge_lists(ball_shot_frames_p1, ball_shot_frames_p2)
        except:
            continue

        # Convert positions to mini court positions
        player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detection, 
                                                                                                            ball_detections,
                                                                                                            court_keypoints)

        player_stats_data = [{
            'frame_num':0,
            'player_1_number_of_shots':0,
            'player_1_total_shot_speed':0,
            'player_1_last_shot_speed':0,
            'player_1_total_player_speed':0,
            'player_1_last_player_speed':0,

            'player_2_number_of_shots':0,
            'player_2_total_shot_speed':0,
            'player_2_last_shot_speed':0,
            'player_2_total_player_speed':0,
            'player_2_last_player_speed':0,
        }]

        mini_court_middle = (mini_court.drawing_key_points[25]+mini_court.drawing_key_points[27])/2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_file = os.path.splitext(video_file)[0]
        for ball_shot_ind in range(len(ball_shot_frames)-1):
            start_frame = ball_shot_frames[ball_shot_ind]
            end_frame = ball_shot_frames[ball_shot_ind+1]
            ball_shot_time_in_seconds = (end_frame-start_frame)/fps 
            if ball_mini_court_detections[start_frame][1][1] < mini_court_middle:
                stroke_player = 2
                opponent = 1
            else:
                stroke_player = 1
                opponent = 2
            if end_frame-start_frame < 10:
                continue
            
            clip_name = str(video_file) + '_' + str(end_frame) + '.mp4' #modified
            clip_path = os.path.join( output_folder+'/'+clip_name)
            out = cv2.VideoWriter(clip_path, fourcc, fps, (video_frames[0].shape[1], video_frames[0].shape[0]))
            for i in range(start_frame, end_frame):
                out.write(video_frames[i])
            out.release()
            #next_file_number+=1

            distance_covered_by_opponent_pixels = player_tracker.calculate_distance(player_mini_court_detections, start_frame, end_frame, opponent)
            distance_covered_by_opponent_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,
                                                                                    constants.DOUBLE_LINE_WIDTH,
                                                                                    mini_court.get_width_of_mini_court()
                                                                                    )
            
            if distance_covered_by_opponent_pixels < mini_court.get_width_of_mini_court()*0.2:
                label=0
            elif distance_covered_by_opponent_pixels < mini_court.get_width_of_mini_court()*0.4:
                label=1
            else:
                label=2
                
            append_video_annotation(clip_name, label, annotation_file_path)

            # Get distance covered by the ball
            distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                            ball_mini_court_detections[end_frame][1])
            distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                            constants.DOUBLE_LINE_WIDTH,
                                                                            mini_court.get_width_of_mini_court()
                                                                            ) 

            # Speed of the ball shot in km/h
            speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

            # player who shot the ball
            player_positions = player_mini_court_detections[start_frame]
            player_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                    ball_mini_court_detections[start_frame][1]))

            # opponent player speed
            opponent_player_id = 1 if player_shot_ball == 2 else 2
            distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                    player_mini_court_detections[end_frame][opponent_player_id])
            distance_covered_by_opponent_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,
                                                                            constants.DOUBLE_LINE_WIDTH,
                                                                            mini_court.get_width_of_mini_court()
                                                                            ) 

            speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

            current_player_stats= deepcopy(player_stats_data[-1])
            current_player_stats['frame_num'] = start_frame
            current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
            current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
            current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

            current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
            current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

            player_stats_data.append(current_player_stats)



        player_stats_data_df = pd.DataFrame(player_stats_data)
        frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
        player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
        player_stats_data_df = player_stats_data_df.ffill()

        player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
        player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
        player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
        player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']



        """ # Draw output
        ## Draw Player Bounding Boxes
        output_video_frames= player_tracker.draw_bboxes(video_frames, player_detection)
        output_video_frames= ball_tracker.draw_bboxes(output_video_frames, ball_detections)

        ## Draw court Keypoints
        output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

        # Draw Mini Court
        output_video_frames = mini_court.draw_mini_court(output_video_frames)
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    

        # Draw Player Stats
        output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)

        ## Draw frame number on top left corner
        for i, frame in enumerate(output_video_frames):
            cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        save_video(output_video_frames, "output_videos/output_video.avi") """

if __name__ == "__main__":
    main()
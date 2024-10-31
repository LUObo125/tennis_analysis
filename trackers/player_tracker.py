from ultralytics import YOLO 
import math
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for i, player_dict in enumerate(player_detections):
            filtered_player_dict={}
            for track_id, bbox in player_dict.items():
                if track_id == chosen_player[0]:
                    filtered_player_dict[1] = bbox
                elif track_id == chosen_player[1]:
                    filtered_player_dict[2] = bbox
            if len(filtered_player_dict) < 2:
                if not 1 in  filtered_player_dict:
                    filtered_player_dict[1] = filtered_player_detections[i-1][1]
                if not 2 in  filtered_player_dict:
                    filtered_player_dict[2] = filtered_player_detections[i-1][2]
            filtered_player_detections.append(filtered_player_dict)
            """ filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict) """
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances_up = []
        distances_down = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in [0, 2, 8, 12, 16, 18, 24, 28]:
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances_up.append((track_id, min_distance))

            min_distance = float('inf')
            for i in [4, 6, 10, 14, 20, 22, 26, 30]:
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances_down.append((track_id, min_distance))
        
        # sorrt the distances in ascending order
        distances_up.sort(key = lambda x: x[1])
        distances_down.sort(key = lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = [distances_down[0][0], distances_up[0][0]]
        return chosen_players


    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True, device=0)[0]
        id_name_dict = results.names

        player_dict = {}
        if results.boxes.id is None:
            return player_dict
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames

    def calculate_distance(self, data, start_frame, end_frame, player_id):
        total_distance = 0
        prev_x1 = None
        prev_y1 = None
        
        for frame_data in data[start_frame - 1: end_frame]:
            if player_id in frame_data:
                x1, y1 = frame_data[player_id]
                if prev_x1 is not None:
                    distance = math.sqrt((x1 - prev_x1)**2 + (y1 - prev_y1)**2)
                    total_distance += distance
                prev_x1 = x1
                prev_y1 = y1
        return total_distance


    
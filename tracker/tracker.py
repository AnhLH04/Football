import cv2
import pandas as pd
from ultralytics import YOLO
import supervision as sv
import os
import pickle
import numpy as np
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detection = self.model.predict(frames[i: i + batch_size])
            detections += detection
        return detections
    def get_tracker(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        detections = self.detect_frames(frames)

        tracks = {
            'players': [],
            'referees': [],
            'ball': []
        }
        for idx, detection in enumerate(detections):
            cls_name = detection.names
            cls_name_inv = {v:k for k,v in cls_name.items()}

            #Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            print(detection_supervision.class_id)
            #Convert GoalKeeper to player object
            for obj_index, class_id in enumerate(detection_supervision.class_id):
                if cls_name[class_id] == 'goalkeeper':
                    detection_supervision.class_id[obj_index] = cls_name_inv['player']

            #Track object
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_name_inv['player']:
                    tracks['players'][idx][track_id] = {'bbox': bbox}
                if cls_id == cls_name_inv['referee']:
                    tracks['referees'][idx][track_id] = {'bbox': bbox}
                if cls_id == cls_name_inv['ball']:
                    tracks['ball'][idx][1] = {'bbox': bbox}
            print(detection_with_tracks)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y = int(bbox[3])
        x = int((bbox[0]+bbox[2])/2)
        width = int(bbox[2]-bbox[0])
        cv2.ellipse(frame,center=(x, y), axes=(width, int(0.3*width)), angle=0.0, startAngle=-45, endAngle=225, color=color, thickness=2)
        if track_id is not None:
            rectangle_w = 40
            rectangle_h = 20
            rectangle_x = x-20
            rectangle_y = y+int(0.3*width)-10
            cv2.rectangle(frame,
                          pt1=(rectangle_x, rectangle_y),
                          pt2=(rectangle_x + rectangle_w, rectangle_y + rectangle_h),
                          color=color,
                          thickness=cv2.FILLED)
            cv2.putText(frame,
                        text=str(track_id),
                        org=(rectangle_x + 10, rectangle_y + 15),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 0),
                        thickness=2)

        return frame
    def draw_triangle(self, frame, bbox, color):
        x = int((bbox[0]+bbox[2])/2)
        y = int(bbox[1])
        points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [points], 0, (0, 0, 0), 2)
        return frame
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for idx, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][idx]
            referee_dict = tracks['referees'][idx]
            ball_dict = tracks['ball'][idx]

            # Draw players annotations
            for track_id, player in player_dict.items():
                color = player.get('team_color', (255, 255, 255))
                frame = self.draw_ellipse(frame, bbox=player['bbox'], color=color, track_id=track_id)
            # Draw referees annotations
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, bbox=referee['bbox'], color=(0, 0, 255))
            output_video_frames.append(frame)
            # Draw ball annotations
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, bbox=ball['bbox'], color=(0, 255, 0))
        return output_video_frames
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        #Interpolate missing value
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1: {'bbox': x}}for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
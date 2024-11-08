import cv2

def read_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        frames.append(frame)
    return frames

def write_video(output_frames, video_output_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output_path, fourcc, 24, (output_frames[0].shape[1], output_frames[0].shape[0]))
    for frame in output_frames:
        out.write(frame)
    out.release()
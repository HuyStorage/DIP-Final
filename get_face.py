import argparse
import numpy as np
import cv2 as cv
import os
from static.utils import *

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

script_dir = 'assets/FaceDetection'

onnx_detection_path = get_file_path("face_detection_yunet_2023mar.onnx")
onnx_recognition_path = get_file_path("face_recognition_sface_2021dec.onnx")

person_name = 'HongNhung'

image_save_dir = os.path.join(script_dir, 'image', person_name)
os.makedirs(image_save_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
args = parser.parse_args()

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
                idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == '__main__':
    detector = cv.FaceDetectorYN.create(
        onnx_detection_path,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )
    recognizer = cv.FaceRecognizerSF.create(onnx_recognition_path, "")

    tm = cv.TickMeter()

    cap = cv.VideoCapture(0)
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    dem = 111
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        faces = detector.detect(frame)  # faces is a tuple
        tm.stop()

        key = cv.waitKey(1)
        if key == 27:  # ESC để thoát
            break

        if key == ord('s') or key == ord('S'):
            if faces[1] is not None:
                face_align = recognizer.alignCrop(frame, faces[1][0])
                file_name = os.path.join(image_save_dir, f'{person_name}_{dem:04d}.bmp')
                cv.imwrite(file_name, face_align)
                print(f"Saved: {file_name}")
                dem += 1

        # Vẽ và hiển thị
        visualize(frame, faces, tm.getFPS())
        cv.imshow('Live', frame)

    cv.destroyAllWindows()

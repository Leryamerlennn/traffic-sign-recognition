import cv2
import os
import numpy as np
from datetime import datetime

from detection.preprocess import preprocessing, resize_frm
from detection.color_segmentation import find_contours
from detection.contour_check import analyze_contours
import tensorflow as tf

MODEL_PATH = 'classification/' + 'model/' + 'traffic_signs_cnn_final.keras'
IMG_SIZE = (32, 32)

CLASS_NAMES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vechiles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vechiles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vechiles over 3.5 metric tons',
    43: 'Unknown',
}

classifier_model = tf.keras.models.load_model(MODEL_PATH)


def classify_sign(bgr_crop: np.ndarray):
    """
    Takes BGR crop of traffic sign, returns (label, confidence).
    Adapt preprocessing to match your training pipeline.
    """
    if bgr_crop.size == 0:
        return None, None
    img = cv2.resize(bgr_crop, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = tf.keras.utils.img_to_array(img)/255.0
    inp = inp[None, ...]
    preds = classifier_model.predict(inp, verbose=0)
    class_id = np.argmax(preds)
    conf = float(preds[0][class_id])
    label = CLASS_NAMES.get(class_id, f"class_{class_id}")

    return label, conf


def camera_work():
    video = cv2.VideoCapture('main4.mp4') # You may choose camera source instead of video file

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps is None or fps == 0:
        fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("output.mp4", fourcc, fps, (640, 640))

    if not video.isOpened():
        print("Error: The camera is not connected")
        return 

    print("The camera is connected. Press the ESC key to exit")
    
    tm = cv2.TickMeter()
    tm.start()
    
    fps = 0
    count = 0
    margin = 5
    run_n = 1
    n = 0
    corr_detections = [0, 0]
    corr_all = 0 
    while True:
        ret, frame = video.read()
        if not ret: 
            print("Error: The frame was not read.")
            break 

        # --- detection pipeline ---
        preproc = preprocessing(frame.copy())
        result1 = find_contours(preproc)
        analysis_result = analyze_contours(result1['contours_result'], (640, 640))

        # --- classification + drawing on original frame ---
        frame_vis = resize_frm(frame, target_size=(640, 640))
        frame_check = frame_vis.copy()
        for cnt in analysis_result:
            x, y, w, h = cv2.boundingRect(cnt)

            x0 = max(0, x - margin)
            y0 = max(0, y - margin)
            x1 = min(frame_check.shape[1], x + w + margin)
            y1 = min(frame_check.shape[0], y + h + margin)

            if x1 <= x0 or y1 <= y0:
                continue

            crop = frame_check[y0:y1, x0:x1]
            crop = cv2.resize(crop, (32, 32))
            label, conf = classify_sign(crop)
            if label == 'Unknown' or conf < 0.9:
                continue

            if label == 'No vehicles':
                corr_detections[0] += 1
                corr_all += 1
            elif label == "Children crossing":
                corr_detections[1] += 1
                corr_all += 1

            cv2.rectangle(frame_vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            text_org = (x0, max(0, y0 - 10))
            cv2.putText(
                frame_vis,
                text,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        writer.write(frame_vis)
        # Exit if needed
        if cv2.waitKey(1) & 0xFF == 27:
            break

        count += 1

    video.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Accuracy of class No vehicles: {corr_detections[0]/count:.2f}, Accuracy of class Children crossing: {corr_detections[1]/count:.2f}, Overall accuracy: {corr_all/(count*2):.2f}")


if __name__ == "__main__":
    camera_work()

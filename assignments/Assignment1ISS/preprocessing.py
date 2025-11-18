import cv2
import os
import numpy as np
from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector


# dataset = load_dataset("imagefolder", data_dir = "./DiffusionFER/DiffusionEmotion_S/Cropped")["train"]
# labels_df = pd.read_csv("./DiffusionFER/DiffusionEmotion_S/dataset_sheet.csv")


device = "cuda"  # or "cpu"

det = FaceDetector("./weights/Alignment_RetinaFace.pth", device=device)
lmk = LandmarkDetector("./weights/Landmark_98.pkl", device=device, device_ids=[0])

# helper function to get landmarks in order to train a landmark->emotion model
def detect_face_and_get_landmarks(image):
    # transform PIL image to BGR numpy array
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    _, dets = det.get_face(img_bgr)
    if dets is None or len(dets) == 0:
        image.save("no_face_detected.png")
        return None

    pts = lmk.detect_landmarks(img_bgr, [dets[0]])[0].astype(np.float32)  # (98,2)
    return pts

def get_label_from_filename(example):
    import re
    filename = os.path.basename(example["image"].filename)
    # match the last number before the file extension
    match = re.search(r'_(\d+)\.', filename)
    if match:
        label = int(match.group(1))
        example["label"] = label
    else:
        # mark invalid or skip
        example["label"] = -1
    return example

def data_preprocessing(dataset):
    dataset = dataset.map(get_label_from_filename)

    images = dataset["image"]
    labels = dataset["label"]

    landmark_features = []
    skipped_indices = []

    for i, img in enumerate(images):
        pts = detect_face_and_get_landmarks(img)
        if pts is None:
            skipped_indices.append(i)
            continue
        landmark_features.append(pts.flatten())

    images = [img for i, img in enumerate(images) if i not in skipped_indices]
    labels = [lbl for i, lbl in enumerate(labels) if i not in skipped_indices]
    landmark_features = [feat for i, feat in enumerate(landmark_features)]
    return landmark_features, labels
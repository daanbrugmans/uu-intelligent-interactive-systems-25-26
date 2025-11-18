import cv2
from matplotlib import pyplot as plt


def draw_faces_on_image(
    image_path: str, determinants: list, confidence_threshold: float = 0.90
) -> None:
    """Taken directly from IIS Lab 1 code and altered.

    Draws bounding boxes on the image at `image_path` using the information in `determinants`.

    Args:
        image_path (str): path to the image.
        determinants (list): information on the faces present in the image at `image_path`.
    """

    img_bgr = cv2.imread(image_path)
    h, w = img_bgr.shape[:2]
    vis = img_bgr.copy()

    boxes = []
    for i, det in enumerate(determinants):
        # Expected RetinaFace-style: [x1,y1,x2,y2,score, lmk(10)?]
        x1, y1, x2, y2 = map(int, det[:4])
        score = float(det[4]) if len(det) > 4 else None

        if score < confidence_threshold:
            continue

        # draw box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # draw 5 landmarks if present
        if len(det) >= 15:
            lm = det[5:15].reshape(-1, 2)
            for lx, ly in lm.astype(int):
                cv2.circle(vis, (lx, ly), 2, (255, 0, 0), -1)

        # label with confidence
        if score is not None:
            cv2.putText(
                vis,
                f"{score:.3f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"{score:.3f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Show in notebook (RGB)
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Faces: {len(determinants)}")
    plt.show()

    # Print details
    print("Bounding boxes [x1,y1,x2,y2]:", boxes)
    for i, det in enumerate(determinants):
        score = float(det[4])
        if score >= confidence_threshold:
            info = {"score": float(det[4]) if len(det) > 4 else None}
            if len(det) >= 15:
                info["landmarks_5"] = det[5:15].reshape(-1, 2).tolist()
            print(f"Face {i + 1}:", info)

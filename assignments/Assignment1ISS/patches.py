import os

import cv2
import numpy as np
import torch


def override_preprocess_image(
    self, image_path: tuple[str, np.ndarray], resize: float = 1.0
) -> tuple[torch.tensor, cv2.MatLike]:
    """Taken from the IIS Lab 1 notebook.

    Acts as an override for OpenFace's FaceDetector so that it can directly accept a NumPy array as an input.
    This prevents having to save a NumPy array as an image to disk before OpenFace can access it.

    Args:
        image_path (tuple[str, np.ndarray]): A string representing the path to an image of a numpy array representation of an image
        resize (float, optional): Resizes the input image with this factor. Defaults to 1.0.

    Raises:
        ValueError: when an image could not be read at the given image_path.
        TypeError: when image_path is not a str or a NumPy array.
        ValueError: when the image's channel count is not 1, 3, or 4.
        ValueError: when the image's shape is not HxW or HxWxC.

    Returns:
        tuple[torch.tensor, cv2.MatLike]: the image as a Torch Tensor and as a cv2 array.
    """
    if isinstance(image_path, (str, os.PathLike)):
        img_raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)  # BGR, 3 channels
        if img_raw is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
    elif isinstance(image_path, np.ndarray):
        img_raw = image_path
    else:
        raise TypeError(
            "image_path must be a str/Path-like path or a numpy.ndarray (BGR frame)."
        )

    if img_raw.ndim == 2:
        # Grayscale -> BGR
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
    elif img_raw.ndim == 3:
        if img_raw.shape[2] == 4:
            # BGRA -> BGR
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2BGR)
        elif img_raw.shape[2] != 3:
            raise ValueError(
                f"Unsupported channel count: {img_raw.shape[2]} (expected 1, 3, or 4)"
            )
    else:
        raise ValueError(
            f"Unsupported image shape {img_raw.shape}; expected HxW or HxWxC."
        )

    # --- Preprocess as in original code
    img = img_raw.astype(np.float32, copy=False)
    if resize != 1.0:
        img = cv2.resize(
            img, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR
        )

    # Mean subtraction in BGR (matching many Caffe-style models)
    img -= (104.0, 117.0, 123.0)

    # Ensure contiguous before transpose (safer with slices or unusual strides)
    img = np.ascontiguousarray(img.transpose(2, 0, 1))  # (C, H, W)

    img = torch.from_numpy(img).unsqueeze(0).to(self.device)  # (1, C, H, W)

    return img, img_raw

import math
from typing import Optional

import cv2
import numpy as np

# Face side smaller than this (in original image pixels) is considered too small
# to be useful, larger faces saturate the size term of the quality score.
REFERENCE_FACE_SIZE = 80.0
# Laplacian variance scale of the sharpness saturation curve.
SHARPNESS_SCALE = 100.0


def _to_gray(crop: np.ndarray) -> np.ndarray:
    """
    Convert an image crop to single channel grayscale.

    Args:
        crop (np.ndarray): The image crop, either grayscale, BGR or BGRA.

    Returns:
        np.ndarray: A single channel grayscale crop.
    """
    if crop.ndim == 2:
        return crop
    if crop.shape[2] == 4:
        return cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def compute_quality(image: np.ndarray,
                    bbox,
                    landmarks: Optional[np.ndarray] = None,
                    det_score: float = 1.0) -> dict:
    """
    Compute deterministic quality heuristics for a single detected face.

    Brightness and sharpness are measured on the face region cropped from the
    original (non resized) image. Pose angles are estimated from the 5 point
    landmarks. The aggregated `quality_score` is a weighted sum of the detection
    score, sharpness, brightness, face size and pose, clipped to [0, 1].

    Args:
        image (np.ndarray): The original image the face was detected on.
        bbox: The face bounding box as (x1, y1, x2, y2) in original image coordinates.
        landmarks (np.ndarray): The 5 point face landmarks, shaped (5, 2). Defaults to None.
        det_score (float): The detector confidence score. Defaults to 1.0.

    Returns:
        dict: A dictionary with `brightness`, `sharpness`, `roll`, `yaw`, `pitch`
            and `quality_score` keys, all plain Python floats.
    """
    brightness = 0.0
    sharpness = 0.0
    roll = 0.0
    yaw = 0.0
    pitch = 0.0

    image_height, image_width = image.shape[:2]
    left, top, right, bottom = (float(v) for v in np.asarray(bbox).ravel()[:4])

    x1, y1 = max(0, int(math.floor(left))), max(0, int(math.floor(top)))
    x2 = min(image_width, int(math.ceil(right)))
    y2 = min(image_height, int(math.ceil(bottom)))
    crop = image[y1:y2, x1:x2]
    if crop.size:
        gray = _to_gray(crop)
        brightness = float(np.clip(np.mean(gray) / 255.0, 0.0, 1.0))
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = float(np.clip(1.0 - math.exp(-laplacian_variance / SHARPNESS_SCALE), 0.0, 1.0))

    if landmarks is not None and np.asarray(landmarks).shape == (5, 2):
        points = np.asarray(landmarks, dtype=np.float64)
        eyes_midpoint = (points[0] + points[1]) / 2.0
        mouth_midpoint = (points[3] + points[4]) / 2.0
        eye_delta = points[1] - points[0]
        eye_distance = max(float(np.linalg.norm(eye_delta)), 1e-6)
        roll = float(np.degrees(np.arctan2(eye_delta[1], eye_delta[0])))
        yaw = float((points[2, 0] - eyes_midpoint[0]) / eye_distance * 90.0)
        vertical = max(float(mouth_midpoint[1] - eyes_midpoint[1]), 1e-6)
        pitch = float(((points[2, 1] - eyes_midpoint[1]) / vertical - 0.45) * 90.0)

    face_width = max(0.0, right - left)
    face_height = max(0.0, bottom - top)
    size_score = float(np.clip(min(face_width, face_height) / REFERENCE_FACE_SIZE, 0.0, 1.0))
    brightness_score = float(np.clip(1.0 - abs(brightness - 0.5) / 0.5, 0.0, 1.0))
    pose_score = float(np.clip(1.0 - max(abs(yaw), abs(pitch), abs(roll)) / 90.0, 0.0, 1.0))

    quality_score = float(
        np.clip(
            0.35 * float(det_score)
            + 0.25 * sharpness
            + 0.15 * brightness_score
            + 0.15 * size_score
            + 0.10 * pose_score,
            0.0,
            1.0,
        )
    )

    return dict(
        brightness=brightness,
        sharpness=sharpness,
        roll=roll,
        yaw=yaw,
        pitch=pitch,
        quality_score=quality_score,
    )

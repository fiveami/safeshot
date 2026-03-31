import os
import cv2


# Optional MTCNN detector (better accuracy for difficult faces)
_mtcnn_detector = None


def load_mtcnn():
    global _mtcnn_detector
    if _mtcnn_detector is not None:
        return _mtcnn_detector
    try:
        from mtcnn.mtcnn import MTCNN

        _mtcnn_detector = MTCNN()
        return _mtcnn_detector
    except Exception:
        _mtcnn_detector = None
        return None


def merge_boxes_iou(boxes, iou_thresh=0.35):
    """Merge overlapping boxes using IoU clustering. Boxes are (x1,y1,x2,y2,score)."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    merged = []
    taken = [False] * len(boxes)
    for i, b in enumerate(boxes):
        if taken[i]:
            continue
        x1, y1, x2, y2, _ = b
        group = [b]
        taken[i] = True
        for j in range(i + 1, len(boxes)):
            if taken[j]:
                continue
            x1b, y1b, x2b, y2b, _ = boxes[j]
            xx1 = max(x1, x1b)
            yy1 = max(y1, y1b)
            xx2 = min(x2, x2b)
            yy2 = min(y2, y2b)
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            area1 = max(1, (x2 - x1) * (y2 - y1))
            area2 = max(1, (x2b - x1b) * (y2b - y1b))
            union = area1 + area2 - inter
            iou = inter / union if union > 0 else 0
            # Merge if IoU is high or one box mostly contains the other.
            if iou > iou_thresh or (inter / area2 > 0.7) or (inter / area1 > 0.7):
                group.append(boxes[j])
                taken[j] = True
        mx1 = min([g[0] for g in group])
        my1 = min([g[1] for g in group])
        mx2 = max([g[2] for g in group])
        my2 = max([g[3] for g in group])
        ms = max([g[4] for g in group])
        merged.append((mx1, my1, mx2, my2, ms))
    return merged


def apply_clahe_bgr(img_bgr, clip=2.0, tile=(8, 8)):
    """Apply CLAHE to BGR image to improve contrast in varied lighting."""
    try:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    except Exception:
        return img_bgr


# Module-level cache for the DNN face detector
_dnn_net = None


def load_dnn_net():
    global _dnn_net
    if _dnn_net is not None:
        return _dnn_net

    base = os.path.dirname(__file__)
    model_file = os.path.join(base, "models", "res10_300x300_ssd_iter_140000.caffemodel")
    config_file = os.path.join(base, "models", "deploy.prototxt")

    if not (os.path.exists(model_file) and os.path.exists(config_file)):
        print("DNN face detector model files not found in 'models/'. Attempting to download them...")
        try:
            download_dnn_models()
        except Exception as exc:
            print(f"Failed to download DNN models: {exc}; falling back to Haar cascades.")
            _dnn_net = None
            return None

    try:
        net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _dnn_net = net
        return _dnn_net
    except Exception as exc:
        print(f"Failed to load DNN face detector: {exc}")
        _dnn_net = None
        return None


def get_aggressiveness_params(level: str):
    """Return parameter tweaks for 'low'|'medium'|'high' aggressiveness."""
    lvl = (level or "medium").lower()
    if lvl == "low":
        return {
            "dnn_conf": 0.40,
            "mtcnn_conf": 0.85,
            "dnn_scales": [1.0],
            "clahe_clip": 1.8,
            "dilation_mult": 0.85,
        }
    if lvl == "high":
        return {
            "dnn_conf": 0.18,
            "mtcnn_conf": 0.60,
            "dnn_scales": [1.0, 1.5, 2.0],
            "clahe_clip": 2.6,
            "dilation_mult": 1.15,
        }
    # medium/default
    return {
        "dnn_conf": 0.25,
        "mtcnn_conf": 0.70,
        "dnn_scales": [1.0, 1.5],
        "clahe_clip": 2.0,
        "dilation_mult": 1.0,
    }


def download_dnn_models():
    """Download required DNN face detector files into models/."""
    import urllib.request

    base = os.path.dirname(__file__)
    models_dir = os.path.join(base, "models")
    os.makedirs(models_dir, exist_ok=True)

    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/res10_300x300_ssd_iter_140000.caffemodel"

    prototxt_dst = os.path.join(models_dir, "deploy.prototxt")
    caffemodel_dst = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    print(f"Downloading deploy.prototxt -> {prototxt_dst}")
    urllib.request.urlretrieve(prototxt_url, prototxt_dst)

    print(f"Downloading res10_300x300_ssd_iter_140000.caffemodel -> {caffemodel_dst}")
    urllib.request.urlretrieve(caffemodel_url, caffemodel_dst)

    if not (os.path.exists(prototxt_dst) and os.path.exists(caffemodel_dst)):
        raise RuntimeError("Failed to download DNN model files")

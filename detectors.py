import os
import cv2
import numpy as np


# Optional MTCNN detector (better accuracy for difficult faces)
_mtcnn_detector = None

def _clip_box(x1, y1, x2, y2, w, h):
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)


def detect_faces_dnn(img_bgr, params, diagnostics=False):
    net = load_dnn_net()
    if net is None:
        return []

    h_img, w_img = img_bgr.shape[:2]
    min_dim = min(w_img, h_img)
    scales = []
    for s in params.get("dnn_scales", [1.0]):
        if s == 1.0 or min_dim * s < 1600:
            scales.append(s)

    boxes = []
    max_side = 1600
    for s in scales:
        sw = int(w_img * s)
        sh = int(h_img * s)
        if max(sw, sh) > max_side:
            adj = max_side / max(sw, sh)
            sw = int(sw * adj)
            sh = int(sh * adj)
            s = s * adj

        resized = cv2.resize(img_bgr, (sw, sh))
        resized_enh = apply_clahe_bgr(resized, clip=params.get("clahe_clip", 2.0), tile=(8, 8))
        blob = cv2.dnn.blobFromImage(resized_enh, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence <= params.get("dnn_conf", 0.25):
                continue
            box = detections[0, 0, i, 3:7] * np.array([sw, sh, sw, sh])
            x1s, y1s, x2s, y2s = box.astype("int")
            x1 = max(0, int(x1s / s))
            y1 = max(0, int(y1s / s))
            x2 = min(w_img - 1, int(x2s / s))
            y2 = min(h_img - 1, int(y2s / s))
            wbox = max(1, x2 - x1)
            hbox = max(1, y2 - y1)
            expand_w = int(wbox * (0.22 if wbox < 80 else 0.14))
            expand_h = int(hbox * (0.27 if hbox < 80 else 0.18))
            x1, y1, x2, y2 = _clip_box(x1 - expand_w, y1 - expand_h, x2 + expand_w, y2 + expand_h, w_img, h_img)
            boxes.append((x1, y1, x2, y2, confidence))

        if diagnostics:
            print(f"DNN scale {s:.2f}: found {len(boxes)} faces so far")

    if len(boxes) < 2 and max(w_img, h_img) > 1000:
        tile_size = 800
        stride = tile_size // 2
        for y in range(0, h_img, stride):
            for x in range(0, w_img, stride):
                if x + tile_size > w_img:
                    x0 = w_img - tile_size
                else:
                    x0 = x
                if y + tile_size > h_img:
                    y0 = h_img - tile_size
                else:
                    y0 = y
                if x0 < 0:
                    x0 = 0
                if y0 < 0:
                    y0 = 0
                patch = img_bgr[y0 : y0 + tile_size, x0 : x0 + tile_size]
                patch_enh = apply_clahe_bgr(patch, clip=params.get("clahe_clip", 2.0), tile=(8, 8))
                blob = cv2.dnn.blobFromImage(patch_enh, 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()
                for i in range(detections.shape[2]):
                    confidence = float(detections[0, 0, i, 2])
                    if confidence <= params.get("dnn_conf", 0.25):
                        continue
                    box = detections[0, 0, i, 3:7] * np.array([tile_size, tile_size, tile_size, tile_size])
                    x1s, y1s, x2s, y2s = box.astype("int")
                    x1 = max(0, x0 + int(x1s))
                    y1 = max(0, y0 + int(y1s))
                    x2 = min(w_img - 1, x0 + int(x2s))
                    y2 = min(h_img - 1, y0 + int(y2s))
                    wbox = max(1, x2 - x1)
                    hbox = max(1, y2 - y1)
                    expand_w = int(wbox * 0.22)
                    expand_h = int(hbox * 0.27)
                    x1, y1, x2, y2 = _clip_box(x1 - expand_w, y1 - expand_h, x2 + expand_w, y2 + expand_h, w_img, h_img)
                    boxes.append((x1, y1, x2, y2, confidence))
        if diagnostics:
            print(f"DNN tile pass added {len(boxes)} total candidate faces")

    return boxes


def detect_faces_mtcnn(img_bgr, params, diagnostics=False):
    mtcnn = load_mtcnn()
    if mtcnn is None:
        return []

    h_img, w_img = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = []
    mt_scales = [1.0]
    if min(w_img, h_img) < 700:
        mt_scales.append(1.5)
    for s in mt_scales:
        target = img_rgb if s == 1.0 else cv2.resize(img_rgb, (int(w_img * s), int(h_img * s)))
        try:
            target_bgr = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
            target_bgr = apply_clahe_bgr(target_bgr, clip=params.get("clahe_clip", 2.0), tile=(8, 8))
            target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        results = mtcnn.detect_faces(target)
        for r in results:
            conf = float(r.get("confidence", 0.0))
            if conf < params.get("mtcnn_conf", 0.70):
                continue
            x, y, w, h = r["box"]
            if s != 1.0:
                x = int(x / s)
                y = int(y / s)
                w = int(w / s)
                h = int(h / s)
            x1 = max(0, int(x - w * 0.12))
            y1 = max(0, int(y - h * 0.18))
            x2 = min(w_img - 1, int(x + w + w * 0.12))
            y2 = min(h_img - 1, int(y + h + h * 0.18))
            boxes.append((x1, y1, x2, y2, conf))
        if diagnostics:
            print(f"MTCNN scale {s:.1f}: found {len(boxes)} faces so far")
    return boxes


def detect_faces_haar(gray, params, detections_only=True):
    h_img, w_img = gray.shape[:2]
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    alt_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    if face_cascade.empty() and alt_face_cascade.empty() and profile_cascade.empty():
        return []

    min_side = min(h_img, w_img)
    min_sz = max(24, int(min_side * 0.025))
    boxes = []

    def _scan(cascade, img, mirrored=False, scale_factor=1.0):
        results = cascade.detectMultiScale(img, scaleFactor=1.08, minNeighbors=4, minSize=(min_sz, min_sz))
        for (x, y, w, h) in results:
            if mirrored:
                x = int((w_img * scale_factor) - (x + w))
                x = max(0, x)
            x1, y1, x2, y2 = _clip_box(x, y, x + w, y + h, w_img, h_img)
            if w < 80:
                padding = int(w * 0.18)
            else:
                padding = int(w * 0.12)
            x1, y1, x2, y2 = _clip_box(x1 - padding, y1 - padding, x2 + padding, y2 + padding, w_img, h_img)
            boxes.append((x1, y1, x2, y2, 0.0))

    if not face_cascade.empty():
        _scan(face_cascade, gray)
    if not alt_face_cascade.empty():
        _scan(alt_face_cascade, gray)
    if not profile_cascade.empty():
        _scan(profile_cascade, gray)
        flipped = cv2.flip(gray, 1)
        _scan(profile_cascade, flipped, mirrored=True)

    return boxes


def estimate_head_priors(gray, existing_boxes, params):
    h_img, w_img = gray.shape[:2]
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
    if body_cascade.empty():
        return []

    min_body = max(36, int(min(h_img, w_img) * 0.08))
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(min_body, min_body))
    priors = []
    for (x, y, w, h) in bodies:
        body_box = (x, y, x + w, y + h)
        overlaps_existing = any(
            max(0, min(body_box[2], bx2) - max(body_box[0], bx1)) > 0
            and max(0, min(body_box[3], by2) - max(body_box[1], by1)) > 0
            for (bx1, by1, bx2, by2, _) in existing_boxes
        )
        if overlaps_existing:
            continue

        head_center_x = x + w // 2
        head_w = max(int(w * 0.46), int(w * 0.38))
        head_h = max(int(h * 0.30), int(w * 0.30))
        x1 = max(0, head_center_x - head_w // 2)
        x2 = min(w_img - 1, head_center_x + head_w // 2)
        y1 = y
        y2 = min(h_img - 1, y + head_h)
        if x2 - x1 >= 40 and y2 - y1 >= 40:
            priors.append((x1, y1, x2, y2))
    return priors


def search_face_priors(gray, priors, params):
    h_img, w_img = gray.shape[:2]
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    if face_cascade.empty() and profile_cascade.empty():
        return []

    boxes = []
    for (x1, y1, x2, y2) in priors:
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        best = None
        for cascade, mirrored in ((face_cascade, False), (profile_cascade, False), (profile_cascade, True)):
            if cascade.empty():
                continue
            if mirrored:
                active_crop = cv2.flip(crop, 1)
            else:
                active_crop = crop
            detections = cascade.detectMultiScale(
                active_crop,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(max(18, int((x2 - x1) * 0.25)), max(18, int((y2 - y1) * 0.25))),
            )
            if len(detections) == 0:
                continue
            candidate = max(detections, key=lambda rect: rect[2] * rect[3])
            if mirrored:
                fx, fy, fw, fh = candidate
                fx = (x2 - x1) - (fx + fw)
            else:
                fx, fy, fw, fh = candidate
            mapped = (x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh)
            mapped = _clip_box(*mapped, w_img, h_img)
            best = mapped
            break

        if best is not None:
            bx1, by1, bx2, by2 = best
            padding = int((bx2 - bx1) * 0.15)
            bx1, by1, bx2, by2 = _clip_box(bx1 - padding, by1 - padding, bx2 + padding, by2 + padding, w_img, h_img)
            boxes.append((bx1, by1, bx2, by2, 0.0))
    return boxes


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

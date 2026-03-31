import sys
from datetime import datetime
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtNetwork import QLocalServer, QLocalSocket
import time
import numpy as np
import cv2
from detectors import (
    apply_clahe_bgr,
    download_dnn_models,
    get_aggressiveness_params,
    load_dnn_net,
    load_mtcnn,
    merge_boxes_iou,
)
from ui import MainWindow


def apply_safe_mode(pixmap: QPixmap, diagnostics: bool = False, aggressiveness: str = "medium") -> QPixmap:
    """
    Take a QPixmap, run Safe Mode (blur faces + top band for names + crop edges),
    and return a new QPixmap.
    """
    try:
        if pixmap.isNull():
            return pixmap

        # Convert QPixmap -> QImage -> numpy array (RGBA)
        image: QImage = pixmap.toImage().convertToFormat(
            QImage.Format.Format_RGBA8888
        )
        width = image.width()
        height = image.height()
        ptr = image.bits()

        # Use bytesPerLine/sizeInBytes where available to safely determine buffer size
        try:
            bpl = int(image.bytesPerLine())
            total = bpl * height
        except Exception:
            # Fallback to sizeInBytes() or byteCount()
            if hasattr(image, "sizeInBytes"):
                total = int(image.sizeInBytes())
            elif hasattr(image, "byteCount"):
                total = int(image.byteCount())
            else:
                total = height * width * 4

        try:
            ptr.setsize(total)
        except Exception:
            # Some PyQt builds provide a read-only buffer; try creating a copy
            buf = bytes(ptr)
            arr = np.frombuffer(buf, dtype=np.uint8)
            arr = arr.reshape((height, width, 4))
        else:
            arr2 = np.frombuffer(ptr, dtype=np.uint8).reshape((height, int(total / height)))
            arr = arr2[:, : width * 4].reshape((height, width, 4))

        # Convert RGBA -> BGR for OpenCV
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

        # We'll collect face boxes from DNN and Haar and blur the union
        (h_img, w_img) = img_bgr.shape[:2]
        face_boxes = []
        dnn_boxes = []
        mtcnn_boxes = []
        haar_boxes = []

        params = get_aggressiveness_params(aggressiveness)
        if diagnostics:
            print(f"Diagnostics enabled. Aggressiveness={aggressiveness}. Params={params}")

        # 1) Multiscale DNN SSD detection to improve small/blurry face recall
        net = load_dnn_net()
        dnn_found = 0
        if net is not None:
            min_dim = min(w_img, h_img)
            # use aggressiveness-driven scales, but filter by image size
            scales = []
            for s in params.get("dnn_scales", [1.0]):
                if s == 1.0:
                    scales.append(1.0)
                else:
                    if min_dim * s < 1600:
                        scales.append(s)

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
                # enhance contrast for difficult lighting conditions before DNN
                resized_enh = apply_clahe_bgr(resized, clip=params.get("clahe_clip", 2.0), tile=(8, 8))
                blob = cv2.dnn.blobFromImage(resized_enh, 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()
                found_scale = 0
                for i in range(0, detections.shape[2]):
                    confidence = float(detections[0, 0, i, 2])
                    # use a slightly lower threshold to improve recall; upscaled images may need a lower cutoff
                    if confidence > params.get("dnn_conf", 0.25):
                        box = detections[0, 0, i, 3:7] * np.array([sw, sh, sw, sh])
                        (x1s, y1s, x2s, y2s) = box.astype("int")
                        # map back to original image coordinates
                        x1 = max(0, int(x1s / s))
                        y1 = max(0, int(y1s / s))
                        x2 = min(w_img - 1, int(x2s / s))
                        y2 = min(h_img - 1, int(y2s / s))
                        # adaptive expansion: expand small detections proportionally more
                        wbox = max(1, x2 - x1)
                        hbox = max(1, y2 - y1)
                        scale_factor = 0.22 if wbox < 80 else 0.14
                        expand_w = int(wbox * scale_factor)
                        expand_h = int(hbox * (scale_factor + 0.05))
                        x1 = max(0, x1 - expand_w)
                        y1 = max(0, y1 - expand_h)
                        x2 = min(w_img - 1, x2 + expand_w)
                        y2 = min(h_img - 1, y2 + expand_h)
                        face_boxes.append((x1, y1, x2, y2, confidence))
                        dnn_boxes.append((x1, y1, x2, y2, confidence))
                        found_scale += 1
                if found_scale > 0:
                    dnn_found += found_scale
                    print(f"DNN scale {s:.2f}: found {found_scale} faces")
            if diagnostics:
                print(f"DNN total found across scales: {dnn_found}")

        # 2) Try MTCNN on original and an upscale when the image is small
        mtcnn = load_mtcnn()
        mt_found_total = 0
        if mtcnn is not None:
            try:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mt_scales = [1.0]
                if min(w_img, h_img) < 600:
                    mt_scales.append(1.5)
                for s in mt_scales:
                    if s == 1.0:
                        target = img_rgb
                    else:
                        tw = int(w_img * s)
                        th = int(h_img * s)
                        target = cv2.resize(img_rgb, (tw, th))
                    # enhance lighting for MTCNN by applying CLAHE on BGR then back to RGB
                    try:
                        target_bgr = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
                        target_bgr = apply_clahe_bgr(target_bgr, clip=params.get("clahe_clip", 2.0), tile=(8, 8))
                        target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
                    except Exception:
                        pass
                    results = mtcnn.detect_faces(target)
                    for r in results:
                        conf = float(r.get("confidence", 0.0))
                        if conf > params.get("mtcnn_conf", 0.70):
                            x, y, w, h = r["box"]
                            if s != 1.0:
                                x = int(x / s)
                                y = int(y / s)
                                w = int(w / s)
                                h = int(h / s)
                            x1 = max(0, int(x - w * 0.14))
                            y1 = max(0, int(y - h * 0.2))
                            x2 = min(w_img - 1, int(x + w + w * 0.14))
                            y2 = min(h_img - 1, int(y + h + h * 0.2))
                            face_boxes.append((x1, y1, x2, y2, conf))
                            mtcnn_boxes.append((x1, y1, x2, y2, conf))
                            mt_found_total += 1
                if mt_found_total > 0:
                    print(f"MTCNN: found {mt_found_total} faces")
                if diagnostics:
                    print(f"MTCNN found: {mt_found_total}")
            except Exception as exc:
                print(f"MTCNN detection failed: {exc}")

        # Haar fallback (also run even if DNN found some to catch missed small faces)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # improve contrast for Haar cascade using CLAHE on the gray image
        try:
            clahe = cv2.createCLAHE(clipLimit=params.get("clahe_clip", 2.0), tileGridSize=(8, 8))
            gray_enh = clahe.apply(gray)
        except Exception:
            gray_enh = gray
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if not face_cascade.empty():
            # dynamic minSize so Haar scales on large images but still picks up small faces
            min_side = min(h_img, w_img)
            min_sz = max(20, int(min_side * 0.02))
            haar_faces = face_cascade.detectMultiScale(
                gray_enh, scaleFactor=1.1, minNeighbors=3, minSize=(min_sz, min_sz)
            )
            for (x, y, w, h) in haar_faces:
                x1, y1, x2, y2 = x, y, x + w, y + h
                # slightly expand; smaller faces get larger expansion
                if w < 80:
                    ew = int(w * 0.18)
                    eh = int(h * 0.22)
                else:
                    ew = int(w * 0.12)
                    eh = int(h * 0.18)
                x1 = max(0, x1 - ew)
                y1 = max(0, y1 - eh)
                x2 = min(w_img - 1, x2 + ew)
                y2 = min(h_img - 1, y2 + eh)
                face_boxes.append((x1, y1, x2, y2, 0.0))
                haar_boxes.append((x1, y1, x2, y2, 0.0))

            # If the image is small, try an upscaled Haar pass to catch tiny faces
            if min(h_img, w_img) < 600:
                up = cv2.resize(gray_enh, (int(w_img * 1.5), int(h_img * 1.5)))
                haar_up = face_cascade.detectMultiScale(up, scaleFactor=1.1, minNeighbors=3, minSize=(18, 18))
                for (x, y, w, h) in haar_up:
                    # map back to original coords
                    x1 = max(0, int(x / 1.5))
                    y1 = max(0, int(y / 1.5))
                    x2 = min(w_img - 1, int((x + w) / 1.5))
                    y2 = min(h_img - 1, int((y + h) / 1.5))
                    ew = int(((x2 - x1) * 0.18))
                    eh = int(((y2 - y1) * 0.22))
                    x1 = max(0, x1 - ew)
                    y1 = max(0, y1 - eh)
                    x2 = min(w_img - 1, x2 + ew)
                    y2 = min(h_img - 1, y2 + eh)
                    face_boxes.append((x1, y1, x2, y2, 0.0))
                    haar_boxes.append((x1, y1, x2, y2, 0.0))
            if diagnostics:
                print(f"Haar found: {len(haar_boxes)} boxes (incl upscaled pass)")

        # Merge overlapping boxes using IoU merging helper, then build tighter per-face elliptical masks
        merged_with_scores = merge_boxes_iou(face_boxes, iou_thresh=0.35)

        merged = []
        if merged_with_scores:
            # prepare per-box meta for distance-based caps
            boxes_meta = []
            for (x1, y1, x2, y2, score, *rest) in merged_with_scores:
                wbox = max(1, x2 - x1)
                hbox = max(1, y2 - y1)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                boxes_meta.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": wbox, "h": hbox, "cx": cx, "cy": cy, "s": score})

            # compute nearest neighbor distances for each box center
            centers = [(b["cx"], b["cy"]) for b in boxes_meta]
            dists = []
            for i, (cx, cy) in enumerate(centers):
                min_d = float("inf")
                for j, (cx2, cy2) in enumerate(centers):
                    if i == j:
                        continue
                    dd = ((cx - cx2) ** 2 + (cy - cy2) ** 2) ** 0.5
                    if dd < min_d:
                        min_d = dd
                if min_d == float("inf"):
                    min_d = max(w_img, h_img)
                dists.append(min_d)

            # number of faces influences aggressiveness
            face_n = len(boxes_meta)
            if face_n <= 2:
                mult = 1.0
            elif face_n <= 4:
                mult = 0.85
            else:
                mult = 0.65

            # apply global aggressiveness multiplier
            mult = mult * params.get("dilation_mult", 1.0)

            # build initial dilations per box, capped by several heuristics
            dils = []
            global_cap = int(min(h_img, w_img) * 0.22)
            for i, b in enumerate(boxes_meta):
                wbox = b["w"]
                hbox = b["h"]
                base = int(max(wbox, hbox) * 0.45)  # more conservative base than before
                base = max(base, 10)
                by_dist = int(dists[i] * 0.35)
                cap = int(1.2 * max(wbox, hbox))
                dil = int(min(base * mult, by_dist, cap, global_cap))
                dil = max(10, min(dil, max(40, int(min(h_img, w_img) * 0.25))))
                dils.append(dil)

            # reduce pairwise overlaps by shrinking dils where needed
            for _ in range(3):
                changed = False
                for i in range(len(boxes_meta)):
                    for j in range(i + 1, len(boxes_meta)):
                        bi = boxes_meta[i]
                        bj = boxes_meta[j]
                        ri = int(bi["w"] / 2 + dils[i])
                        rj = int(bj["w"] / 2 + dils[j])
                        # estimate overlap if ellipses overlap using center distance
                        dd = ((bi["cx"] - bj["cx"]) ** 2 + (bi["cy"] - bj["cy"]) ** 2) ** 0.5
                        if dd < (ri + rj) * 0.85 and dd > 0:
                            # too close — reduce both proportionally
                            new_di = max(8, int(dils[i] * 0.80))
                            new_dj = max(8, int(dils[j] * 0.80))
                            if new_di < dils[i] or new_dj < dils[j]:
                                dils[i] = new_di
                                dils[j] = new_dj
                                changed = True
                if not changed:
                    break

            # helper to build mask from dils
            def _build_mask_from_dils(dils_list):
                mask = np.zeros((h_img, w_img), dtype=np.uint8)
                for i, b in enumerate(boxes_meta):
                    cx = b["cx"]
                    cy = b["cy"]
                    rx = int(b["w"] / 2 + dils_list[i])
                    ry = int(b["h"] / 2 + dils_list[i] * 0.9)
                    rx = min(rx, w_img)
                    ry = min(ry, h_img)
                    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                mask = cv2.medianBlur(mask, 5)
                return mask

            mask = _build_mask_from_dils(dils)

            # if mask covers too much of the image (over 50%), shrink dils iteratively
            frac = (mask > 0).sum() / float(mask.size)
            attempts = 0
            while frac > 0.50 and attempts < 6:
                dils = [max(8, int(d * 0.85)) for d in dils]
                mask = _build_mask_from_dils(dils)
                frac = (mask > 0).sum() / float(mask.size)
                attempts += 1

            # Extract contours as final blurred regions
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                nx1 = max(0, x)
                ny1 = max(0, y)
                nx2 = min(w_img - 1, x + w)
                ny2 = min(h_img - 1, y + h)
                merged.append((nx1, ny1, nx2, ny2))
            if len(merged) > 0:
                print(f"Merged/dilated to {len(merged)} face regions from {len(face_boxes)} detections (mask frac={frac:.2f})")

            # If diagnostics enabled, produce an overlay image with raw detections and mask
            if diagnostics:
                try:
                    dbg = img_bgr.copy()
                    # draw raw boxes: DNN red, MTCNN green, Haar blue
                    for (x1, y1, x2, y2, s) in dnn_boxes:
                        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    for (x1, y1, x2, y2, s) in mtcnn_boxes:
                        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    for (x1, y1, x2, y2, s) in haar_boxes:
                        cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    # draw merged contours as thick yellow outlines
                    for (x1, y1, x2, y2) in merged:
                        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    # overlay mask semi-transparently
                    mask_col = cv2.merge([mask // 2, mask // 2, np.zeros_like(mask)])
                    dbg = cv2.addWeighted(dbg, 0.8, mask_col, 0.6, 0)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    diag_fn = f"diagnostics_{ts}.png"
                    cv2.imwrite(diag_fn, dbg)
                    print(f"Saved diagnostics image: {diag_fn}")
                except Exception as exc:
                    print(f"Failed to generate diagnostics image: {exc}")
        else:
            # No merged mask produced — optionally save raw detection overlay when diagnostics is enabled
            if diagnostics and (dnn_boxes or mtcnn_boxes or haar_boxes):
                try:
                    dbg = img_bgr.copy()
                    for (x1, y1, x2, y2, s) in dnn_boxes:
                        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    for (x1, y1, x2, y2, s) in mtcnn_boxes:
                        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    for (x1, y1, x2, y2, s) in haar_boxes:
                        cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    diag_fn = f"diagnostics_{ts}.png"
                    cv2.imwrite(diag_fn, dbg)
                    print(f"Saved diagnostics image (raw detections): {diag_fn}")
                except Exception as exc:
                    print(f"Failed to generate raw diagnostics image: {exc}")

        # Apply blur to merged boxes
        faces_blurred = 0
        for (x1, y1, x2, y2) in merged:
            if x2 - x1 > 8 and y2 - y1 > 8:
                kw = ((x2 - x1) // 3) | 1
                kh = ((y2 - y1) // 3) | 1
                kw = max(15, kw)
                kh = max(15, kh)
                roi = img_bgr[y1:y2, x1:x2]
                # Apply stronger blur for larger regions, but avoid blurring across face boundaries
                roi = cv2.GaussianBlur(roi, (kw, kh), 0)
                img_bgr[y1:y2, x1:x2] = roi
                faces_blurred += 1
        if faces_blurred > 0:
            print(f"Blurred {faces_blurred} face regions (merged)")

        # 2) Blur a horizontal band at the top to hide names / headers
        h_img, w_img, _ = img_bgr.shape
        band_height = int(h_img * 0.12)  # top 12% of image
        text_regions_blurred = 0
        if band_height > 0:
            header_roi = img_bgr[0:band_height, :]
            # Simple text-region detection in the top band
            try:
                gray_top = cv2.cvtColor(header_roi, cv2.COLOR_BGR2GRAY)
                # increase contrast
                gray_top = cv2.equalizeHist(gray_top)
                thr = cv2.adaptiveThreshold(
                    gray_top, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 9
                )
                # Morphological close to join letters
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
                closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w > 30 and h > 8 and w / max(1, h) > 2.0:
                        # expand a bit and blur that region in the header_roi
                        ex = int(w * 0.12)
                        ey = int(h * 0.2)
                        x1 = max(0, x - ex)
                        y1 = max(0, y - ey)
                        x2 = min(header_roi.shape[1] - 1, x + w + ex)
                        y2 = min(header_roi.shape[0] - 1, y + h + ey)
                        sub = header_roi[y1:y2, x1:x2]
                        kx = ((x2 - x1) // 2) | 1
                        ky = ((y2 - y1) // 2) | 1
                        kx = max(11, kx)
                        ky = max(5, ky)
                        sub = cv2.GaussianBlur(sub, (kx, ky), 0)
                        header_roi[y1:y2, x1:x2] = sub
                        text_regions_blurred += 1
                if text_regions_blurred == 0:
                    # No text regions found, fallback to blurring entire top band
                    header_roi = cv2.GaussianBlur(header_roi, (41, 41), 25)
                    text_regions_blurred = 1
                img_bgr[0:band_height, :] = header_roi
            except Exception as exc:
                print(f"Top-band text detection failed: {exc}")
                img_bgr[0:band_height, :] = cv2.GaussianBlur(header_roi, (41, 41), 25)
        if text_regions_blurred > 0:
            print(f"Blurred {text_regions_blurred} text regions in top band")

        # 3) Slight edge crop to remove rough borders
        border = max(4, int(min(h_img, w_img) * 0.01))  # 1% border, at least 4px
        if h_img > 2 * border and w_img > 2 * border:
            img_bgr = img_bgr[border : h_img - border, border : w_img - border]

        # Convert back BGR -> RGBA
        img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
        h2, w2, ch2 = img_rgba.shape
        bytes_per_line = ch2 * w2
        qimage_out = QImage(
            img_rgba.data, w2, h2, bytes_per_line, QImage.Format.Format_RGBA8888
        )

        # Make deep copy so data stays valid
        qimage_out = qimage_out.copy()

        return QPixmap.fromImage(qimage_out)
    except Exception as exc:
        # Don't crash the whole app for image processing errors
        print(f"Safe mode processing failed: {exc}")
        return pixmap


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Single-instance IPC: if another snip is running, tell it to close
    server_name = "SafeShotSnipSocket"

    def handle_incoming():
        sock = server.nextPendingConnection()
        if sock is None:
            return
        # Wait briefly for data and read
        if sock.waitForReadyRead(500):
            data = bytes(sock.readAll()).decode(errors="ignore")
            if "CLOSE" in data:
                # Ask the running instance to quit
                app.quit()
        sock.disconnectFromServer()

    server = QLocalServer()
    # If we can listen, we're the primary server for this session
    if server.listen(server_name):
        server.newConnection.connect(handle_incoming)
    else:
        # Try to send a CLOSE command to the existing instance
        sock = QLocalSocket()
        sock.connectToServer(server_name)
        if sock.waitForConnected(500):
            sock.write(b"CLOSE")
            sock.flush()
            sock.waitForBytesWritten(500)
            sock.disconnectFromServer()
        else:
            # Possibly a stale server socket — remove it so we can listen
            try:
                QLocalServer.removeServer(server_name)
            except Exception:
                pass

        # Try to become the server now; retry a few times because the
        # previous instance may need a moment to fully exit and release
        # the server name on the OS.
        became_server = False
        for _ in range(8):
            if server.listen(server_name):
                server.newConnection.connect(handle_incoming)
                became_server = True
                break
            try:
                QLocalServer.removeServer(server_name)
            except Exception:
                pass
            time.sleep(0.15)

        # If we still didn't become the server, continue anyway; the
        # overlay will still run but we won't accept IPC requests.
        if not became_server:
            print(
                "Warning: could not become IPC server; single-instance control may not work"
            )

    window = MainWindow(
        safe_mode_fn=apply_safe_mode,
        download_models_fn=download_dnn_models,
        load_dnn_net_fn=load_dnn_net,
    )
    window.show()

    sys.exit(app.exec())

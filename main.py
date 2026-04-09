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
    detect_faces_dnn,
    detect_faces_haar,
    detect_faces_mtcnn,
    download_dnn_models,
    estimate_head_priors,
    get_aggressiveness_params,
    load_dnn_net,
    merge_boxes_iou,
    search_face_priors,
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

        # We'll collect face boxes from DNN and MTCNN first, then apply narrow fallbacks.
        (h_img, w_img) = img_bgr.shape[:2]
        face_boxes = []
        dnn_boxes = []
        mtcnn_boxes = []
        haar_boxes = []

        params = get_aggressiveness_params(aggressiveness)
        if diagnostics:
            print(f"Diagnostics enabled. Aggressiveness={aggressiveness}. Params={params}")

        dnn_boxes = detect_faces_dnn(img_bgr, params, diagnostics=diagnostics)
        face_boxes.extend(dnn_boxes)
        mtcnn_boxes = detect_faces_mtcnn(img_bgr, params, diagnostics=diagnostics)
        face_boxes.extend(mtcnn_boxes)

        if diagnostics:
            print(f"DNN found {len(dnn_boxes)}, MTCNN found {len(mtcnn_boxes)}")

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        try:
            clahe = cv2.createCLAHE(clipLimit=params.get("clahe_clip", 2.0), tileGridSize=(8, 8))
            gray_enh = clahe.apply(gray)
        except Exception:
            gray_enh = gray

        if len(face_boxes) < 4:
            priors = estimate_head_priors(gray_enh, face_boxes, params)
            if diagnostics:
                print(f"Head priors from body search: {len(priors)}")
            prior_faces = search_face_priors(gray_enh, priors, params)
            if prior_faces:
                face_boxes.extend(prior_faces)
                haar_boxes.extend(prior_faces)
                if diagnostics:
                    print(f"Head prior face search found {len(prior_faces)} faces")

        if len(face_boxes) < 3:
            full_haar = detect_faces_haar(gray_enh, params)
            if full_haar:
                face_boxes.extend(full_haar)
                haar_boxes.extend(full_haar)
                if diagnostics:
                    print(f"Full Haar fallback found {len(full_haar)} faces")

        if diagnostics:
            print(f"Total face candidates before merge: {len(face_boxes)}")

        # Merge only heavily overlapping detections and keep nearby faces separate
        merged_with_scores = merge_boxes_iou(face_boxes, iou_thresh=0.45)

        merged = []
        if merged_with_scores:
            # prepare per-box meta for tighter per-face padding
            boxes_meta = []
            for (x1, y1, x2, y2, score, *rest) in merged_with_scores:
                wbox = max(1, x2 - x1)
                hbox = max(1, y2 - y1)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                boxes_meta.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": wbox, "h": hbox, "cx": cx, "cy": cy, "s": score})

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
                dists.append(max(min_d, max(boxes_meta[i]["w"], boxes_meta[i]["h"]) * 0.75))

            for i, b in enumerate(boxes_meta):
                pad = int(max(b["w"], b["h"]) * 0.18)
                if dists[i] < max(b["w"], b["h"]) * 1.2:
                    pad = max(5, int(pad * 0.7))
                pad = min(pad, int(min(h_img, w_img) * 0.15))
                x1 = max(0, b["x1"] - pad)
                y1 = max(0, b["y1"] - pad)
                x2 = min(w_img - 1, b["x2"] + pad)
                y2 = min(h_img - 1, b["y2"] + pad)
                if x2 - x1 > 8 and y2 - y1 > 8:
                    merged.append((x1, y1, x2, y2))

            if diagnostics:
                print(f"Generated {len(merged)} final face regions from {len(boxes_meta)} merged candidates")

            # If diagnostics enabled, produce an overlay image with raw detections and final boxes
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

from datetime import datetime
from typing import Callable

from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QMouseEvent, QPaintEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class SnipWindow(QWidget):
    def __init__(
        self,
        on_done=None,
        diagnostics: bool = False,
        aggressiveness: str = "Medium",
        safe_mode_fn: Callable[[QPixmap, bool, str], QPixmap] | None = None,
    ) -> None:
        super().__init__()

        # Full-screen, frameless overlay
        self.setWindowTitle("SafeShot - Snip")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)

        # Capture the desktop once and use it as the background
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            raise RuntimeError("No screen available")

        # Grab entire desktop (window 0)
        self.desktop_pixmap: QPixmap = screen.grabWindow(0)

        # Resize to the current geometry if necessary (handles scaling)
        self.showFullScreen()

        # Drag start/end points
        self.begin: QPoint | None = None
        self.end: QPoint | None = None

        # Callback invoked when snip completes or is cancelled
        # Signature: on_done(pixmap: QPixmap|None, filename: str|None)
        self.on_done = on_done
        self._done_called = False
        self._safe_mode_fn = safe_mode_fn
        # Snapshot of diagnostics/aggressiveness settings from the caller
        self._diagnostics = bool(diagnostics)
        self._aggressiveness = aggressiveness or "Medium"

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.begin = event.pos()
            self.end = self.begin
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.begin is not None:
            self.end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.begin is not None:
            self.end = event.pos()
            rect = QRect(self.begin, self.end).normalized()

            MIN_PIXEL_THRESHOLD = 10
            if rect.width() >= MIN_PIXEL_THRESHOLD and rect.height() >= MIN_PIXEL_THRESHOLD:
                try:
                    try:
                        scale = float(self.desktop_pixmap.devicePixelRatioF())
                    except Exception:
                        scale = float(self.desktop_pixmap.devicePixelRatio())

                    src_rect = QRect(
                        int(rect.left() * scale),
                        int(rect.top() * scale),
                        int(rect.width() * scale),
                        int(rect.height() * scale),
                    )

                    cropped = self.desktop_pixmap.copy(src_rect)
                    safe_mode_fn = self._safe_mode_fn or (lambda p, **_kwargs: p)
                    safe_pixmap = safe_mode_fn(
                        cropped, diagnostics=self._diagnostics, aggressiveness=self._aggressiveness
                    )

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{ts}.png"
                    saved = safe_pixmap.save(filename)
                    if saved:
                        print(f"Saved screenshot (safe mode): {filename}")
                    else:
                        print("Failed to save screenshot to file")

                    QApplication.clipboard().setPixmap(safe_pixmap)
                    print("Copied safe screenshot to clipboard")

                    if callable(self.on_done) and not self._done_called:
                        try:
                            self.on_done(safe_pixmap, filename)
                        finally:
                            self._done_called = True
                except Exception as exc:
                    print(f"Error during Safe Snip processing: {exc}")
                    if callable(self.on_done) and not self._done_called:
                        try:
                            self.on_done(None, None)
                        finally:
                            self._done_called = True

            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            print("Snip canceled")
            if callable(self.on_done) and not self._done_called:
                try:
                    self.on_done(None, None)
                finally:
                    self._done_called = True
            self.close()

    def closeEvent(self, event):
        if callable(self.on_done) and not self._done_called:
            try:
                self.on_done(None, None)
            finally:
                self._done_called = True
        super().closeEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self.desktop_pixmap.isNull():
            qp.drawPixmap(self.rect(), self.desktop_pixmap, self.desktop_pixmap.rect())

        overlay_color = QColor(0, 0, 0, 120)
        qp.fillRect(self.rect(), overlay_color)

        if self.begin is not None and self.end is not None:
            sel_rect = QRect(self.begin, self.end).normalized()
            try:
                scale = float(self.desktop_pixmap.devicePixelRatioF())
            except Exception:
                scale = float(self.desktop_pixmap.devicePixelRatio())

            src_rect = QRect(
                int(sel_rect.left() * scale),
                int(sel_rect.top() * scale),
                int(sel_rect.width() * scale),
                int(sel_rect.height() * scale),
            )

            qp.drawPixmap(sel_rect, self.desktop_pixmap, src_rect)
            qp.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.SolidLine))
            qp.drawRect(sel_rect)


class MainWindow(QWidget):
    def __init__(
        self,
        safe_mode_fn: Callable[[QPixmap, bool, str], QPixmap],
        download_models_fn: Callable[[], None],
        load_dnn_net_fn: Callable[[], object | None],
    ) -> None:
        super().__init__()
        self.setWindowTitle("SafeShot")
        self._safe_mode_fn = safe_mode_fn
        self._download_models_fn = download_models_fn
        self._load_dnn_net_fn = load_dnn_net_fn

        self.setStyleSheet(
            """
            QWidget {
                background-color: #05060A;
                color: #F5F5F5;
                font-family: "Segoe UI", sans-serif;
                font-size: 10pt;
            }
            QPushButton {
                background-color: #1f6feb;
                border-radius: 6px;
                padding: 6px 10px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #2568d8;
            }
            QPushButton:pressed {
                background-color: #1a5cc4;
            }
            QPushButton:disabled {
                background-color: #333333;
                color: #888888;
            }
            QLabel {
                color: #d0d7de;
            }
            """
        )

        self.setFixedSize(360, 260)

        self.last_pixmap: QPixmap | None = None
        self.last_filename: str | None = None

        self.new_button = QPushButton("New")
        self.new_button.clicked.connect(self.start_snip)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_last)
        self.save_button.setEnabled(False)
        self.download_models_button = QPushButton("Models")
        self.download_models_button.clicked.connect(self.download_models)
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(QApplication.instance().quit)
        self.diagnostics_button = QPushButton("Diagnostics: Off")
        self._diagnostics_enabled = False

        def _toggle_diag():
            self._diagnostics_enabled = not self._diagnostics_enabled
            self.diagnostics_button.setText("Diagnostics: On" if self._diagnostics_enabled else "Diagnostics: Off")

        self.diagnostics_button.clicked.connect(_toggle_diag)

        self._aggressiveness_level = "Medium"
        self.aggressiveness_button = QPushButton("Aggressiveness: Medium")

        def _cycle_aggressiveness():
            order = ["Low", "Medium", "High"]
            cur = self._aggressiveness_level
            idx = order.index(cur) if cur in order else 1
            idx = (idx + 1) % len(order)
            self._aggressiveness_level = order[idx]
            self.aggressiveness_button.setText(f"Aggressiveness: {self._aggressiveness_level}")

        self.aggressiveness_button.clicked.connect(_cycle_aggressiveness)

        small_btn_style = (
            "background-color: #1f6feb; border-radius: 6px; padding: 4px 8px; "
            "color: #ffffff; font-size: 9pt; min-width: 68px;"
        )
        for btn in (self.new_button, self.save_button, self.download_models_button, self.quit_button):
            btn.setFixedHeight(30)
            btn.setStyleSheet(small_btn_style)
        self.diagnostics_button.setFixedHeight(30)
        self.diagnostics_button.setStyleSheet(small_btn_style)
        self.aggressiveness_button.setFixedHeight(30)
        self.aggressiveness_button.setStyleSheet(small_btn_style)

        self.info_label = QLabel("Screenshots are automatically blurred for safety.")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.thumb_label = QLabel("No screenshot yet")
        self.thumb_label.setFixedSize(320, 140)
        self.thumb_label.setStyleSheet(
            "background: #111827; color: #9CA3AF; border-radius: 8px; padding: 6px;"
        )
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        toolbar.addWidget(self.new_button)
        toolbar.addWidget(self.save_button)
        toolbar.addWidget(self.diagnostics_button)
        toolbar.addWidget(self.aggressiveness_button)
        toolbar.addWidget(self.download_models_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.quit_button)

        layout.addLayout(toolbar)
        layout.addWidget(self.thumb_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)
        self.setLayout(layout)

    def start_snip(self):
        self.hide()

        def on_done(pixmap, filename):
            if isinstance(pixmap, QPixmap):
                self.last_pixmap = pixmap
                self.last_filename = filename
                thumb = pixmap.scaled(
                    self.thumb_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.thumb_label.setPixmap(thumb)
                self.save_button.setEnabled(True)

            self.show()
            try:
                self._snip_window = None
            except Exception:
                pass

        try:
            self._snip_window = SnipWindow(
                on_done=on_done,
                diagnostics=self._diagnostics_enabled,
                aggressiveness=self._aggressiveness_level,
                safe_mode_fn=self._safe_mode_fn,
            )
            self._snip_window.show()
        except Exception as exc:
            print(f"Failed to start snip: {exc}")
            self.show()

    def save_last(self):
        if self.last_pixmap is None:
            return
        fn, _ = QFileDialog.getSaveFileName(
            self,
            "Save screenshot",
            self.last_filename or "screenshot.png",
            "PNG Files (*.png);;All Files (*)",
        )
        if fn:
            self.last_pixmap.save(fn)

    def download_models(self):
        self.download_models_button.setEnabled(False)
        self.info_label.setText("Downloading DNN models... this may take a moment.")
        QApplication.processEvents()
        try:
            self._download_models_fn()
            _ = self._load_dnn_net_fn()
            self.info_label.setText("DNN models downloaded and loaded successfully.")
            print("DNN models downloaded and loaded successfully")
        except Exception as exc:
            self.info_label.setText(f"Failed to download models: {exc}")
            print(f"Failed to download models: {exc}")
        finally:
            self.download_models_button.setEnabled(True)

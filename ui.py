from datetime import datetime
from typing import Callable

from PyQt6.QtCore import QPoint, QRect, Qt, QSize, QTimer
from PyQt6.QtGui import QColor, QGuiApplication, QIcon, QMouseEvent, QPaintEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget


class SnipWindow(QWidget):
    def __init__(
        self,
        on_done=None,
        diagnostics: bool = False,
        aggressiveness: str = "Medium",
        safe_mode_fn: Callable[[QPixmap, bool, str], QPixmap] | None = None,
        desktop_pixmap: QPixmap | None = None,
    ) -> None:
        super().__init__()

        # Full-screen, frameless overlay
        self.setWindowTitle("SafeShot - Snip")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)

        # Use the provided desktop capture if available, otherwise grab it now.
        if desktop_pixmap is not None and not desktop_pixmap.isNull():
            self.desktop_pixmap = desktop_pixmap
        else:
            screen = QGuiApplication.primaryScreen()
            if screen is None:
                raise RuntimeError("No screen available")
            self.desktop_pixmap = screen.grabWindow(0)

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
                except Exception as exc:
                    print(f"Error during Safe Snip processing: {exc}")

                # Call on_done and close after successful snip
                if callable(self.on_done) and not self._done_called:
                    self.on_done(safe_pixmap, filename)
                    self._done_called = True
                self.close()
            else:
                # Rectangle too small, reset for another attempt
                self.begin = None
                self.end = None
                self.update()

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


def make_symbol_icon(symbol_type: str, size: int = 18, color: QColor | None = None) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(color or QColor(255, 255, 255, 200), 2)
    painter.setPen(pen)

    if symbol_type == "new":
        painter.drawLine(int(size * 0.2), int(size * 0.5), int(size * 0.8), int(size * 0.5))
        painter.drawLine(int(size * 0.5), int(size * 0.2), int(size * 0.5), int(size * 0.8))
    elif symbol_type == "save":
        painter.drawRect(int(size * 0.2), int(size * 0.18), int(size * 0.6), int(size * 0.64))
        painter.drawRect(int(size * 0.35), int(size * 0.28), int(size * 0.3), int(size * 0.2))
        painter.drawLine(int(size * 0.2), int(size * 0.62), int(size * 0.8), int(size * 0.62))
    elif symbol_type == "models":
        painter.drawRect(int(size * 0.16), int(size * 0.16), int(size * 0.26), int(size * 0.26))
        painter.drawRect(int(size * 0.58), int(size * 0.16), int(size * 0.26), int(size * 0.26))
        painter.drawRect(int(size * 0.16), int(size * 0.58), int(size * 0.26), int(size * 0.26))
        painter.drawRect(int(size * 0.58), int(size * 0.58), int(size * 0.26), int(size * 0.26))
    elif symbol_type == "quit":
        painter.drawLine(int(size * 0.25), int(size * 0.25), int(size * 0.75), int(size * 0.75))
        painter.drawLine(int(size * 0.25), int(size * 0.75), int(size * 0.75), int(size * 0.25))
    elif symbol_type == "diagnostics":
        painter.drawLine(int(size * 0.14), int(size * 0.6), int(size * 0.35), int(size * 0.35))
        painter.drawLine(int(size * 0.35), int(size * 0.35), int(size * 0.5), int(size * 0.55))
        painter.drawLine(int(size * 0.5), int(size * 0.55), int(size * 0.65), int(size * 0.3))
        painter.drawLine(int(size * 0.65), int(size * 0.3), int(size * 0.86), int(size * 0.6))
    elif symbol_type == "aggressiveness":
        painter.drawLine(int(size * 0.28), int(size * 0.7), int(size * 0.28), int(size * 0.4))
        painter.drawLine(int(size * 0.48), int(size * 0.7), int(size * 0.48), int(size * 0.3))
        painter.drawLine(int(size * 0.68), int(size * 0.7), int(size * 0.68), int(size * 0.5))
    painter.end()
    return QIcon(pixmap)


class MainWindow(QWidget):
    def __init__(
        self,
        safe_mode_fn: Callable[[QPixmap, bool, str], QPixmap],
        download_models_fn: Callable[[], None],
        load_dnn_net_fn: Callable[[], object | None],
    ) -> None:
        super().__init__()
        self.setWindowTitle("SafeShot")
        self.setObjectName("root")
        self._safe_mode_fn = safe_mode_fn
        self._download_models_fn = download_models_fn
        self._load_dnn_net_fn = load_dnn_net_fn

        self.setStyleSheet(
            """
            QWidget#root {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #11151F, stop: 1 #171D2F);
                color: #F5F5F5;
                font-family: "Segoe UI", "Segoe UI Variable", sans-serif;
                font-size: 10pt;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 24px;
                min-width: 500px;
            }
            QPushButton {
                background-color: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 14px;
                color: #F8F8FF;
                padding: 10px 14px;
                min-height: 36px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.14);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.10);
            }
            QPushButton:disabled {
                background-color: rgba(255, 255, 255, 0.04);
                color: rgba(255, 255, 255, 0.5);
                border-color: rgba(255, 255, 255, 0.06);
            }
            QPushButton#accent {
                background-color: #0A6DF5;
                border: none;
                color: #FFFFFF;
            }
            QPushButton#accent:hover {
                background-color: #2F7CFF;
            }
            QPushButton#destructive {
                background-color: rgba(255, 75, 90, 0.16);
                border: 1px solid rgba(255, 75, 90, 0.35);
                color: #FF7A85;
            }
            QPushButton#destructive:hover {
                background-color: rgba(255, 75, 90, 0.22);
            }
            QLabel#title {
                font-size: 13pt;
                font-weight: 700;
                color: #FFFFFF;
            }
            QLabel#subtitle {
                font-size: 9.5pt;
                color: #A8B0C4;
            }
            QLabel#thumb {
                background-color: rgba(255, 255, 255, 0.05);
                color: #B0B9C7;
                border-radius: 18px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            QLabel#info {
                color: #A8B0C4;
                font-size: 9.5pt;
            }
            QWidget#card {
                background-color: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 24px;
            }
            """
        )

        self.setMinimumSize(500, 380)
        self.resize(500, 380)

        self.last_pixmap: QPixmap | None = None
        self.last_filename: str | None = None

        self.title_label = QLabel("SafeShot")
        self.title_label.setObjectName("title")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.subtitle_label = QLabel("Instant blurred screenshots, safe by default.")
        self.subtitle_label.setObjectName("subtitle")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setWordWrap(True)

        self.new_button = QPushButton("New")
        self.new_button.setObjectName("accent")
        self.new_button.setIcon(make_symbol_icon("new"))
        self.new_button.clicked.connect(self.start_snip)
        self.save_button = QPushButton("Save")
        self.save_button.setIcon(make_symbol_icon("save"))
        self.save_button.clicked.connect(self.save_last)
        self.save_button.setEnabled(False)
        self.download_models_button = QPushButton("Models")
        self.download_models_button.setIcon(make_symbol_icon("models"))
        self.download_models_button.clicked.connect(self.download_models)
        self.quit_button = QPushButton("Quit")
        self.quit_button.setObjectName("destructive")
        self.quit_button.setIcon(make_symbol_icon("quit"))
        self.quit_button.clicked.connect(QApplication.instance().quit)
        self.diagnostics_button = QPushButton("Diagnostics: Off")
        self.diagnostics_button.setIcon(make_symbol_icon("diagnostics"))
        self._diagnostics_enabled = False

        def _toggle_diag():
            self._diagnostics_enabled = not self._diagnostics_enabled
            self.diagnostics_button.setText("Diagnostics: On" if self._diagnostics_enabled else "Diagnostics: Off")

        self.diagnostics_button.clicked.connect(_toggle_diag)

        self._aggressiveness_level = "Medium"
        self.aggressiveness_button = QPushButton("Aggressiveness: Medium")
        self.aggressiveness_button.setIcon(make_symbol_icon("aggressiveness"))

        def _cycle_aggressiveness():
            order = ["Low", "Medium", "High"]
            cur = self._aggressiveness_level
            idx = order.index(cur) if cur in order else 1
            idx = (idx + 1) % len(order)
            self._aggressiveness_level = order[idx]
            self.aggressiveness_button.setText(f"Aggressiveness: {self._aggressiveness_level}")

        self.aggressiveness_button.clicked.connect(_cycle_aggressiveness)

        base_button_style = (
            "background-color: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.12); "
            "border-radius: 14px; color: #F8F8FF; padding: 8px 12px;"
        )
        for btn in (
            self.new_button,
            self.save_button,
            self.download_models_button,
            self.quit_button,
            self.diagnostics_button,
            self.aggressiveness_button,
        ):
            btn.setFixedHeight(36)
            btn.setMinimumWidth(0)
            btn.setMaximumWidth(220)
            btn.setIconSize(QSize(16, 16))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(base_button_style)

        self.new_button.setMinimumWidth(108)
        self.save_button.setMinimumWidth(96)
        self.download_models_button.setMinimumWidth(96)
        self.quit_button.setMinimumWidth(100)
        self.diagnostics_button.setMinimumWidth(128)
        self.aggressiveness_button.setMinimumWidth(148)

        self.info_label = QLabel("Screenshots are automatically blurred for safety.")
        self.info_label.setObjectName("info")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.thumb_label = QLabel("No screenshot yet")
        self.thumb_label.setObjectName("thumb")
        self.thumb_label.setMinimumSize(340, 160)
        self.thumb_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.subtitle_label)

        toolbar_top = QHBoxLayout()
        toolbar_top.setSpacing(8)
        toolbar_top.addWidget(self.new_button)
        toolbar_top.addWidget(self.save_button)
        toolbar_top.addWidget(self.download_models_button)
        toolbar_top.addStretch(1)
        toolbar_top.addWidget(self.quit_button)

        toolbar_bottom = QHBoxLayout()
        toolbar_bottom.setSpacing(8)
        toolbar_bottom.addWidget(self.diagnostics_button)
        toolbar_bottom.addWidget(self.aggressiveness_button)
        toolbar_bottom.addStretch(1)

        card_widget = QWidget(self)
        card_widget.setObjectName("card")
        card_layout = QVBoxLayout(card_widget)
        card_layout.setContentsMargins(18, 18, 18, 18)
        card_layout.setSpacing(16)
        card_layout.addLayout(header_layout)
        card_layout.addLayout(toolbar_top)
        card_layout.addLayout(toolbar_bottom)
        card_layout.addWidget(self.thumb_label, alignment=Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.info_label)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.addWidget(card_widget)
        self.setLayout(main_layout)

    def start_snip(self):
        self.hide()
        QApplication.processEvents()

        def on_done(pixmap, filename):
            if isinstance(pixmap, QPixmap):
                self.last_pixmap = pixmap
                self.last_filename = filename

                screen = QGuiApplication.primaryScreen()
                max_display_w = 640
                max_display_h = 340
                if screen is not None:
                    geo = screen.availableGeometry()
                    max_display_w = min(max_display_w, int(geo.width() * 0.72))
                    max_display_h = min(max_display_h, int(geo.height() * 0.55))

                thumb = pixmap.scaled(
                    max_display_w,
                    max_display_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.thumb_label.setFixedSize(thumb.size())
                self.thumb_label.setPixmap(thumb)
                self.save_button.setEnabled(True)
                self.setMinimumSize(500, 380)
                self.resize(max(self.width(), thumb.width() + 120), max(self.height(), thumb.height() + 240))

            self.show()
            try:
                self._snip_window = None
            except Exception:
                pass

        def open_snip_overlay():
            try:
                screen = QGuiApplication.primaryScreen()
                desktop_pixmap = QPixmap()
                if screen is not None:
                    desktop_pixmap = screen.grabWindow(0)

                self._snip_window = SnipWindow(
                    on_done=on_done,
                    diagnostics=self._diagnostics_enabled,
                    aggressiveness=self._aggressiveness_level,
                    safe_mode_fn=self._safe_mode_fn,
                    desktop_pixmap=desktop_pixmap,
                )
                self._snip_window.show()
                self._snip_window.raise_()
                self._snip_window.activateWindow()

                # Show the main window behind the overlay once the overlay is active
                self.show()
                self.raise_()
            except Exception as exc:
                print(f"Failed to start snip: {exc}")
                self.show()

        QTimer.singleShot(120, open_snip_overlay)

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

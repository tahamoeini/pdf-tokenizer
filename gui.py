import sys
import os
import json
import threading
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QProgressBar, QTextEdit, QMessageBox, QFrame,
    QCheckBox, QComboBox, QListWidget, QListWidgetItem, QSplitter, QSizePolicy,
    QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QMimeData
from PyQt5.QtGui import QDropEvent, QColor, QFont
from PyQt5.QtCore import QTimer

# Import the extractor module
import extract


class WorkerSignals(QObject):
    """Signals for background processing."""
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    message = pyqtSignal(str)


class DropFrame(QFrame):
    """Custom QFrame that accepts drop events."""
    drop_signal = pyqtSignal(list)

    def dragEnterEvent(self, event):
        """Handle drag enter."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        urls = event.mimeData().urls()
        files = []
        for url in urls:
            path = url.toLocalFile()
            if path.lower().endswith('.pdf'):
                files.append(path)
        if files:
            self.drop_signal.emit(files)


class PDFExtractorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.output_folder = None
        self.worker_signals = WorkerSignals()
        self.pdf_files = []
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("PDF Tokenizer & Structure Extractor")
        self.setGeometry(100, 100, 1100, 720)
        self.setStyleSheet(self.get_stylesheet())

        # Main widget and layout with splitter for two-column layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Left: primary area (drop, file list, actions)
        left_col = QWidget()
        left_layout = QVBoxLayout()

        title = QLabel("PDF Tokenizer & Structure Extractor")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        left_layout.addWidget(title)

        # Prominent drop area
        drop_frame = DropFrame()
        drop_frame.setObjectName("drop_area")
        drop_frame.setMinimumHeight(160)
        drop_label = QLabel("📂 Drag & Drop PDFs here\nor click 'Browse' to select files")
        drop_label.setAlignment(Qt.AlignCenter)
        drop_label.setFont(QFont("Segoe UI", 12))
        df_layout = QVBoxLayout()
        df_layout.addWidget(drop_label)
        drop_frame.setLayout(df_layout)
        drop_frame.drop_signal.connect(self.on_files_dropped)
        left_layout.addWidget(drop_frame)

        # File list widget
        self.file_list = QListWidget()
        self.file_list.setFixedHeight(120)
        left_layout.addWidget(self.file_list)

        # Action buttons row
        action_row = QHBoxLayout()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_pdfs)
        browse_btn.setProperty('primary', True)
        action_row.addWidget(browse_btn)

        extract_btn = QPushButton("Start")
        extract_btn.clicked.connect(self.start_extraction)
        extract_btn.setProperty('accent', True)
        extract_btn.setFixedWidth(120)
        action_row.addWidget(extract_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_files)
        action_row.addWidget(clear_btn)

        open_output_btn = QPushButton("Open Output")
        open_output_btn.clicked.connect(self.open_output_folder)
        action_row.addWidget(open_output_btn)

        left_layout.addLayout(action_row)

        # Progress and log
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        log_label = QLabel("Processing Log")
        left_layout.addWidget(log_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setMinimumHeight(220)
        left_layout.addWidget(self.log_text)

        left_col.setLayout(left_layout)

        # Right: settings and outputs
        right_col = QWidget()
        right_layout = QVBoxLayout()

        settings_title = QLabel("Settings")
        settings_title.setFont(title_font)
        settings_title.setStyleSheet("margin-bottom:8px;")
        right_layout.addWidget(settings_title)

        self.chk_ocr = QCheckBox("Enable OCR fallback")
        self.chk_ocr.setChecked(True)
        right_layout.addWidget(self.chk_ocr)

        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("DPI:"))
        self.spin_dpi = QSpinBox()
        self.spin_dpi.setRange(72, 600)
        self.spin_dpi.setValue(300)
        hlayout.addWidget(self.spin_dpi)
        right_layout.addLayout(hlayout)

        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("OCR Lang:"))
        self.combo_lang = QComboBox()
        self.combo_lang.addItems(["eng", "spa", "fra", "deu"]) 
        self.combo_lang.setCurrentText("eng")
        lang_layout.addWidget(self.combo_lang)
        right_layout.addLayout(lang_layout)

        right_layout.addWidget(QLabel("Outputs"))
        self.outputs_list = QListWidget()
        self.outputs_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.outputs_list.itemActivated.connect(self.open_artifact)
        right_layout.addWidget(self.outputs_list)

        right_col.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_col)
        splitter.addWidget(right_col)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)

    def get_stylesheet(self):
        """Return custom stylesheet."""
        return """
        QMainWindow { background-color: #FAFAFC; }
        QLabel { color: #1F2937; }
        /* Default button visible background for material style */
        QPushButton { background: #52525B; color: white; border-radius: 8px; padding: 8px 12px; border: none; }
        QPushButton[property~="primary"] { background: #6750A4; }
        QPushButton[property~="accent"] { background: #0B8043; }
        QPushButton:pressed { transform: translateY(1px); }
        QPushButton:hover { filter: brightness(1.03); }
        QPushButton:disabled { background: rgba(15,23,42,0.06); color: rgba(15,23,42,0.3); }
        QFrame#drop_area { border: 2px dashed rgba(99,102,241,0.25); border-radius: 12px; background: linear-gradient(180deg, #FFFFFF, #FBFBFF); }
        QListWidget { background: white; border: 1px solid rgba(15,23,42,0.05); border-radius: 8px; }
        QTextEdit { background-color: #FFFFFF; border: 1px solid rgba(15,23,42,0.05); border-radius: 8px; padding: 8px; }
        QProgressBar { border-radius: 8px; height: 18px; background: rgba(15,23,42,0.04); }
        QProgressBar::chunk { background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #6750A4, stop:1 #7F56D9); }
        QSpinBox, QComboBox { padding: 4px; }
        """

    def connect_signals(self):
        """Connect worker signals to slots."""
        self.worker_signals.progress.connect(self.update_progress)
        self.worker_signals.message.connect(self.log_message)
        self.worker_signals.error.connect(self.show_error)
        self.worker_signals.finished.connect(self.extraction_finished)
        # allow double-click on file list to remove
        try:
            self.file_list.itemDoubleClicked.connect(self.remove_file_item)
        except Exception:
            pass

    def on_files_dropped(self, files):
        """Handle files dropped onto the drop area."""
        for file in files:
            if file not in self.pdf_files:
                self.pdf_files.append(file)
                item = QListWidgetItem(Path(file).name)
                item.setData(Qt.UserRole, file)
                self.file_list.addItem(item)
        self.update_file_label()

    def drag_enter_event(self, event):
        """Handle drag enter."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def drop_event(self, event: QDropEvent):
        """Handle drop event."""
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            if path.lower().endswith('.pdf'):
                if path not in self.pdf_files:
                    self.pdf_files.append(path)
        self.update_file_label()

    def browse_pdfs(self):
        """Browse for PDF files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select PDF Files", "", "PDF Files (*.pdf)"
        )
        for file in files:
            if file not in self.pdf_files:
                self.pdf_files.append(file)
                item = QListWidgetItem(Path(file).name)
                item.setData(Qt.UserRole, file)
                self.file_list.addItem(item)
        self.update_file_label()

    def select_output_folder(self):
        """Select output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_label.setText(f"Output: {folder}")

    def update_file_label(self):
        """Update file list label."""
        # update list widget selection summary
        if self.pdf_files:
            count = len(self.pdf_files)
            self.file_list.setToolTip(', '.join([Path(f).name for f in self.pdf_files]))
        else:
            self.file_list.clear()

    def clear_files(self):
        """Clear selected files."""
        self.pdf_files = []
        self.file_list.clear()
        self.outputs_list.clear()
        self.log_text.clear()
        self.progress_bar.setValue(0)

    def start_extraction(self):
        """Start extraction in a background thread."""
        if not self.pdf_files:
            QMessageBox.warning(self, "No Files", "Please select at least one PDF file.")
            return
        self.log_text.clear()
        self.progress_bar.setValue(0)

        # Run extraction in background
        thread = threading.Thread(target=self.run_extraction)
        thread.daemon = True
        thread.start()

    def run_extraction(self):
        """Run extraction."""
        try:
            self.worker_signals.message.emit("Starting extraction...")

            # Override output directory if custom folder is selected
            if self.output_folder:
                original_output = extract.OUTPUT_DIR
                extract.OUTPUT_DIR = self.output_folder
                os.makedirs(self.output_folder, exist_ok=True)
            else:
                original_output = extract.OUTPUT_DIR

            # Copy PDFs to resources folder for processing
            resources_dir = os.path.join(extract.OUTPUT_DIR, "..", "resources")
            os.makedirs(resources_dir, exist_ok=True)

            for idx, pdf_file in enumerate(self.pdf_files):
                self.worker_signals.message.emit(f"Processing: {Path(pdf_file).name}")
                # Copy file to resources
                import shutil
                dest = os.path.join(resources_dir, Path(pdf_file).name)
                shutil.copy(pdf_file, dest)

                # Process
                processed_data = extract.process_pdf(dest)
                if processed_data:
                    self.worker_signals.message.emit(f"✓ Completed: {Path(pdf_file).name}")
                    # Show produced outputs in outputs panel
                    outs = processed_data.get('outputs') or processed_data.get('outputs', [])
                    if isinstance(outs, list):
                        for p in outs:
                            item = QListWidgetItem(p)
                            item.setData(Qt.UserRole, p)
                            self.outputs_list.addItem(item)
                else:
                    self.worker_signals.message.emit(f"✗ Failed: {Path(pdf_file).name}")

                progress = int((idx + 1) / len(self.pdf_files) * 100)
                self.worker_signals.progress.emit(progress)

            # Save stats
            stats_file = os.path.join(extract.OUTPUT_DIR, "processing_stats.json")
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    self.worker_signals.message.emit(f"\n✅ Extraction Complete!")
                    self.worker_signals.message.emit(f"Successful: {stats['successful']}/{stats['total_pdfs']}")
                    self.worker_signals.message.emit(f"Output folder: {extract.OUTPUT_DIR}")

            self.worker_signals.finished.emit()

            # Restore
            if self.output_folder:
                extract.OUTPUT_DIR = original_output

        except Exception as e:
            self.worker_signals.error.emit(str(e))

    def update_progress(self, value):
        """Update progress bar."""
        self.progress_bar.setValue(value)

    def log_message(self, message):
        """Log a message."""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def show_error(self, error_msg):
        """Show error message."""
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg}")
        self.log_text.append(f"❌ Error: {error_msg}")

    def extraction_finished(self):
        """Called when extraction finishes."""
        self.log_message("\n✅ Extraction finished!")
        QMessageBox.information(self, "Success", "PDF extraction completed!\nCheck the output folder for results.")

    def open_output_folder(self):
        """Open output folder in file explorer."""
        output = self.output_folder or os.path.join(os.getcwd(), extract.OUTPUT_DIR)
        if os.path.exists(output):
            os.startfile(output)
        else:
            QMessageBox.warning(self, "Folder Not Found", f"Output folder does not exist: {output}")

    def open_artifact(self, item: QListWidgetItem):
        """Open a produced artifact (file or folder) from outputs panel."""
        path = item.data(Qt.UserRole)
        if not path:
            path = item.text()
        if os.path.exists(path):
            # open file if file, else open folder
            if os.path.isfile(path):
                os.startfile(path)
            else:
                os.startfile(path)
        else:
            QMessageBox.warning(self, "Not found", f"Path not found: {path}")

    def remove_file_item(self, item: QListWidgetItem):
        """Remove a file from the queue on double click."""
        path = item.data(Qt.UserRole)
        if path in self.pdf_files:
            self.pdf_files.remove(path)
        self.file_list.takeItem(self.file_list.row(item))


def main():
    app = QApplication(sys.argv)
    gui = PDFExtractorGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

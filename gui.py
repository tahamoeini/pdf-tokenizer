import sys
import os
import json
import threading
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QProgressBar, QTextEdit, QMessageBox, QFrame
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
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet(self.get_stylesheet())

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # Title
        title = QLabel("PDF Tokenizer & Structure Extractor")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Drag and drop area
        drop_frame = DropFrame()
        drop_frame.setObjectName("drop_area")
        drop_frame.setMinimumHeight(150)
        drop_layout = QVBoxLayout()
        drop_label = QLabel("📁 Drag & Drop PDFs here\nor click 'Browse' to select files")
        drop_label.setAlignment(Qt.AlignCenter)
        drop_label.setFont(QFont("Arial", 12))
        drop_layout.addWidget(drop_label)
        drop_frame.setLayout(drop_layout)
        drop_frame.drop_signal.connect(self.on_files_dropped)
        layout.addWidget(drop_frame)

        # File list label
        self.file_label = QLabel("No files selected")
        layout.addWidget(self.file_label)

        # Buttons layout (row 1)
        button_layout1 = QHBoxLayout()
        browse_btn = QPushButton("📂 Browse PDFs")
        browse_btn.clicked.connect(self.browse_pdfs)
        button_layout1.addWidget(browse_btn)

        output_btn = QPushButton("📁 Select Output Folder")
        output_btn.clicked.connect(self.select_output_folder)
        button_layout1.addWidget(output_btn)

        self.output_label = QLabel("Output: Default (processed_data/)")
        self.output_label.setStyleSheet("color: #666; font-style: italic;")
        button_layout1.addWidget(self.output_label)
        layout.addLayout(button_layout1)

        # Buttons layout (row 2)
        button_layout2 = QHBoxLayout()
        extract_btn = QPushButton("▶ Extract PDFs")
        extract_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        extract_btn.clicked.connect(self.start_extraction)
        button_layout2.addWidget(extract_btn)

        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.clicked.connect(self.clear_files)
        button_layout2.addWidget(clear_btn)

        open_output_btn = QPushButton("📂 Open Output Folder")
        open_output_btn.clicked.connect(self.open_output_folder)
        button_layout2.addWidget(open_output_btn)
        layout.addLayout(button_layout2)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Output log
        log_label = QLabel("Processing Log:")
        layout.addWidget(log_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(250)
        self.log_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.log_text)

        main_widget.setLayout(layout)

    def get_stylesheet(self):
        """Return custom stylesheet."""
        return """
        QMainWindow {
            background-color: #f5f5f5;
        }
        QLabel {
            color: #333;
        }
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1976D2;
        }
        QPushButton:pressed {
            background-color: #1565C0;
        }
        QFrame#drop_area {
            border: 2px dashed #2196F3;
            border-radius: 8px;
            background-color: #e3f2fd;
        }
        QTextEdit {
            background-color: #fff;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 8px;
            font-family: Courier;
        }
        QProgressBar {
            border: 1px solid #ccc;
            border-radius: 4px;
            height: 20px;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
        }
        """

    def connect_signals(self):
        """Connect worker signals to slots."""
        self.worker_signals.progress.connect(self.update_progress)
        self.worker_signals.message.connect(self.log_message)
        self.worker_signals.error.connect(self.show_error)
        self.worker_signals.finished.connect(self.extraction_finished)

    def on_files_dropped(self, files):
        """Handle files dropped onto the drop area."""
        for file in files:
            if file not in self.pdf_files:
                self.pdf_files.append(file)
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
        self.update_file_label()

    def select_output_folder(self):
        """Select output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_label.setText(f"Output: {folder}")

    def update_file_label(self):
        """Update file list label."""
        if self.pdf_files:
            count = len(self.pdf_files)
            self.file_label.setText(f"✓ {count} file(s) selected: {', '.join([Path(f).name for f in self.pdf_files])}")
        else:
            self.file_label.setText("No files selected")

    def clear_files(self):
        """Clear selected files."""
        self.pdf_files = []
        self.update_file_label()
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


def main():
    app = QApplication(sys.argv)
    gui = PDFExtractorGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

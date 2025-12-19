"""
PyInstaller build script to generate exe from gui.py
Usage: python build_exe.py
"""
import os
import PyInstaller.__main__

def build_exe():
    """Build the executable using PyInstaller."""
    print("Building PDF Tokenizer GUI executable...")
    
    pyinstaller_args = [
        'gui.py',
        '--onefile',  # Single exe file
        '--windowed',  # No console window
        '--name=PDFTokenizer',
        '--icon=icon.ico' if os.path.exists('icon.ico') else '',
        '--add-data=extract.py:.',
        '--hidden-import=PyQt5',
        '--hidden-import=extract',
        '--collect-all=nltk',
        '--collect-all=cv2',
    ]
    
    # Filter out empty strings
    pyinstaller_args = [arg for arg in pyinstaller_args if arg]
    
    PyInstaller.__main__.run(pyinstaller_args)
    print("\n✅ Executable built successfully!")
    print("📁 Look in the 'dist' folder for PDFTokenizer.exe")


if __name__ == "__main__":
    build_exe()

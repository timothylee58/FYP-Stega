import binascii
import cv2
import sys
import os
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow, QLineEdit, QVBoxLayout, QWidget, QPushButton, QTextEdit, QHBoxLayout, QProgressBar, QMessageBox, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage,QFont
from PyQt5.QtCore import Qt,QThread, pyqtSignal
from PIL import Image
import pytesseract
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad,pad
import base64
import random
import secrets
import re
import threading
from reedsolo import RSCodec, ReedSolomonError

class WorkerThread(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self, start_frame, end_frame, video_path, password, parent=None):
        super(WorkerThread, self).__init__(parent)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.video_path = video_path
        self.password = password
        
    @staticmethod
    def is_valid_base64_string(s):  
        pattern = r'^[A-Za-z0-9+/]+={0,2}$'
        if re.match(pattern, s):
            return True
        return False
    
    def decrypt_text(self, combined_data, password):
     try:
        # Extract IV and ciphertext
        iv = combined_data[:16]
        ciphertext = combined_data[16:]

        key_length = 32
        iterations = 100000

        # Derive the encryption key using PBKDF2
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), salt=iv, length=key_length, iterations=iterations, backend=default_backend())
        key = kdf.derive(password.encode())

        # Create an AES cipher object for decryption
        cipher = AES.new(key, AES.MODE_CBC, iv)

        # Decrypt the ciphertext
        plaintext = cipher.decrypt(ciphertext)

        # Unpad the plaintext to get the original message
        decrypted_text = unpad(plaintext, AES.block_size).decode('utf-8')

        return decrypted_text
     except Exception as e:
        print(f"Decryption failed: {e}")
        return None
    
    def run(self):
      cap = cv2.VideoCapture(self.video_path)
      if not cap.isOpened():
        self.update_signal.emit("Error: Unable to open video file.")
        return

      for frame_index in range(self.start_frame, self.end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            self.update_signal.emit(f"Frame {frame_index}: Unable to read frame")
            continue

        # Crop to OCR region if necessary, e.g., frame = frame[y:y+h, x:x+w]
        # Preprocess the frame
        preprocessed_frame = self.preprocess_frame(frame)
        
        # Perform OCR on the preprocessed frame
        extracted_text = pytesseract.image_to_string(preprocessed_frame).strip().replace('\n', '')

        # Validate the OCR output
        if not self.is_valid_base64_string(extracted_text):
            self.update_signal.emit(f"Frame {frame_index}: Invalid base64 string")
            continue

        try:
            encrypted_bytes = base64.b64decode(extracted_text)
            decrypted_text = self.decrypt_text(encrypted_bytes, self.password)
            self.update_signal.emit(f"Frame {frame_index}: Decrypted text: {decrypted_text}")
        except (binascii.Error, ValueError, KeyError) as e:
            self.update_signal.emit(f"Frame {frame_index}: Decryption error: {e}")
        except Exception as e:
            self.update_signal.emit(f"Frame {frame_index}: Unexpected error: {e}")

      cap.release()
      self.update_signal.emit("Extraction and decryption complete.")
     
    def preprocess_frame(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Crop the region of interest if necessary, e.g., crop_roi = gray[y:y+h, x:x+w]
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply thresholding to get a binary image
        _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dilate the text to make it more contiguous
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        # Resize the image to make it larger, this can help with OCR accuracy
        scaled = cv2.resize(dilated, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        return scaled
     
    

class VideoTextExtractor(QMainWindow):
    def __init__(self):
         super().__init__()

         self.setWindowTitle("Video Text Extractor")
         self.setGeometry(100, 100, 800, 600)
         self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)

         self.central_widget = QWidget(self)
         self.setCentralWidget(self.central_widget)

         self.layout = QVBoxLayout(self.central_widget)

         self.video_label = QLabel(self)
         self.video_label.setAlignment(Qt.AlignCenter)
         self.layout.addWidget(self.video_label)

         self.extracted_text_label = QTextEdit(self)
         self.extracted_text_label.setReadOnly(True)
         self.layout.addWidget(self.extracted_text_label)
            
         self.progress_bar = QProgressBar(self)
         self.layout.addWidget(self.progress_bar)
         
         self.selected_video_path = QLabel(self)
         self.layout.addWidget(self.selected_video_path)
         
         self.start_frame_entry = QLineEdit(self)
         self.start_frame_entry.setToolTip("Enter the starting frame number for text extraction.")
         self.end_frame_entry = QLineEdit(self)
            
         self.layout.addWidget(QLabel("Start Frame:"))
         self.layout.addWidget(self.start_frame_entry)
     
         self.layout.addWidget(QLabel("End Frame:"))
         self.layout.addWidget(self.end_frame_entry)

        # Decryption key/password entry
         self.decryption_key_entry = QLineEdit(self)
         self.decryption_key_entry.setEchoMode(QLineEdit.Password)  # Set the default mode to Password
         self.layout.addWidget(QLabel("Decryption Key/Password:"))
         self.layout.addWidget(self.decryption_key_entry)

        # Hide/Show button for password
         self.hide_show_btn = QPushButton("Hide/Show", self)
         self.hide_show_btn.clicked.connect(self.toggle_password_visibility)
         self.layout.addWidget(self.hide_show_btn)
        
         self.browse_button = QPushButton("Browse", self)
         self.browse_button.clicked.connect(self.browse_file)
         self.layout.addWidget(self.browse_button)

         self.save_button = QPushButton("Save Text to File", self)
         self.save_button.clicked.connect(self.save_text_to_file)
         self.layout.addWidget(self.save_button)

         self.extract_button = QPushButton("Extraction and Decryption", self)
         self.extract_button.setToolTip("Click to extract and decrypt text from the selected video frames.")
         self.extract_button.clicked.connect(self.extract_text_threaded)
         self.layout.addWidget(self.extract_button)

         self.image_label = QLabel(self)
         self.image_label.setAlignment(Qt.AlignCenter)
         self.layout.addWidget(self.image_label)
            
         self.video_path = ""
         
    def toggle_password_visibility(self):
        # Toggle the visibility of the password
        if self.decryption_key_entry.echoMode() == QLineEdit.Password:
            self.decryption_key_entry.setEchoMode(QLineEdit.Normal)
        else:
            self.decryption_key_entry.setEchoMode(QLineEdit.Password)    
        
    def browse_file(self):
     file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video files (*.mp4 *.avi *.mov);;All Files (*)")
     if file_path:
        self.selected_video_path.setText(file_path)
        self.video_path = file_path

    def start_extraction(self):
        self.worker_thread.start()
            
    def setup_video_capture(self):
     input_path, _ = QFileDialog.getOpenFileName(None, "Open Video File", "", "MP4 files (*.mp4);;All Files (*)")
     if not input_path or not os.path.exists(input_path):
        self.show_error_message("Invalid file path or file does not exist.")
        return False

     self.video_path = input_path
     cap = cv2.VideoCapture(self.video_path)
     if not cap.isOpened():
        self.show_error_message("Error opening video file.")
        return False

     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
     cap.release()
     return True

    def toggle_password_visibility(self):
        # Check if the password is currently hidden
        if self.decryption_key_entry.echoMode() == QLineEdit.Password:
            # Show the password
            self.decryption_key_entry.setEchoMode(QLineEdit.Normal)
        else:
            # Hide the password
            self.decryption_key_entry.setEchoMode(QLineEdit.Password)
      
    def extract_text(self):
       self.setup_video_capture()
       self.process_frames_for_text_extraction()
       input_path, _ = QFileDialog.getOpenFileName(None, "Open Video File", "", "MP4 files (*.mp4);;All Files (*)")
       if not input_path or not os.path.exists(input_path):
         self.show_error_message("Invalid file path or file does not exist.")
         return

       if input_path == "":
         print("File selection cancelled.")
         return

       self.video_path = input_path
       cap = cv2.VideoCapture(self.video_path)
       if not cap.isOpened():
         print("Error opening video file")
         return

       total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       self.frame_selection_slider.setMaximum(total_frames - 1)
     
       try:
         start_frame = int(self.start_frame_entry.text())
         end_frame = int(self.end_frame_entry.text())
       except ValueError:
        print("Please enter valid integer values for start and end frames.")
        cap.release()
        return

       if start_frame < 0 or end_frame > total_frames or start_frame >= end_frame:
        print("Invalid frame range. Please enter a valid range.")
        cap.release()
        return
    
 
       frame_skip = 10  # Process every 10th frame
       total_frames_to_process = (end_frame - start_frame) // frame_skip

       extracted_texts = []

       for frame_index in range(start_frame, end_frame, frame_skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        extracted_text = self.extract_text_from_frame(frame)
        extracted_texts.append((frame_index, extracted_text))

        # Update progress bar less frequently
        if frame_index % (frame_skip * 15) == 0:
            progress = ((frame_index - start_frame) / total_frames_to_process) * 100
            self.progress_bar.setValue(int(progress))

       # Final progress bar update
       self.progress_bar.setValue(100)

       self.display_results(extracted_texts)
       cap.release()
        
    def extract_text_from_frame(self, frame):
     # Preprocess the frame if necessary
     preprocessed_frame = self.preprocess_frame(frame) if hasattr(self, 'preprocess_frame') else frame

    # Example usage of decode_data_from_higher_bits
     for y in range(frame.shape[0]):  # Iterate over rows
        for x in range(frame.shape[1]):  # Iterate over columns
            pixel = frame[y, x]
           
            # Use a lambda function to pass the method
            extracted_data = self.decode_data_from_higher_bits(pixel, lambda p: self.extract_bits_from_pixel(p))

     # Continue with text extraction
     pil_image = Image.fromarray(frame)
     extracted_text = pytesseract.image_to_string(pil_image)
     return extracted_text
    
    def process_frames_for_text_extraction(self):
     if not self.video_path:
        self.show_error_message("No video file selected.")
        return

     cap = cv2.VideoCapture(self.video_path)
     start_frame, end_frame = self.get_frame_range()
     if start_frame is None or end_frame is None:
        cap.release()
        return

     extracted_texts = []
     total_frames_to_process = end_frame - start_frame
     for frame_index in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        extracted_text = self.extract_text_from_frame(frame)
        extracted_texts.append((frame_index, extracted_text))

        progress = ((frame_index - start_frame + 1) / total_frames_to_process) * 100
        self.progress_bar.setValue(int(progress))

     self.progress_bar.setValue(0)
     self.display_results(extracted_texts)
     cap.release()
    
    def get_frame_range(self):
     try:
        start_frame = int(self.start_frame_entry.text())
        end_frame = int(self.end_frame_entry.text())
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError("Invalid frame range.")
        return start_frame, end_frame
     except ValueError as e:
        self.show_error_message(str(e))
        return None, None

    
    def decode_data_from_higher_bits(self,pixel,extract_bits_from_pixel):
      # Extract the bits from the higher bit planes
      extracted_bits = self.extract_bits_from_pixel(pixel)
      return extracted_bits
    
    
    def extract_bits_from_pixel(self, pixel):
         return (pixel >> 8) & 0x0F  # Extracting higher 8 bits 
            
    
    def extract_text_threaded(self):
         # Retrieve parameters from UI
        start_frame = self.getStartFrame()
        end_frame = self.getEndFrame()
        video_path = self.video_path
        password = self.decryption_key_entry.text()

         # Validate inputs
        if self.validateInputs(video_path, start_frame, end_frame, password):
            # Initialize and start the worker thread
            self.worker_thread = WorkerThread(start_frame, end_frame, video_path, password)
            self.worker_thread.update_signal.connect(self.update_method)
            self.worker_thread.start()
            
    def update_method(self, message):
     self.extracted_text_label.append(message)
     

    def getStartFrame(self):
        try:
            return int(self.start_frame_entry.text())
        except ValueError:
            self.show_error_message("Invalid start frame.")
            return None

    def getEndFrame(self):
        try:
            return int(self.end_frame_entry.text())
        except ValueError:
            self.show_error_message("Invalid end frame.")
            return None

    def validateInputs(self, video_path, start_frame, end_frame, password):
        if not os.path.exists(video_path):
            self.show_error_message("Invalid video path.")
            return False

        if start_frame is None or end_frame is None or start_frame < 0 or end_frame < start_frame:
            self.show_error_message("Invalid frame range.")
            return False

        if not password:
            self.show_error_message("Password cannot be empty.")
            return False

        # Optionally, check if the frame range is within the video's frame count
        if not self.isFrameRangeValid(video_path, start_frame, end_frame):
            self.show_error_message("Frame range is out of bounds for the video.")
            return False

        return True

    def isFrameRangeValid(self, video_path, start_frame, end_frame):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return start_frame >= 0 and end_frame <= total_frames

    def preprocess_frame(self, frame):
    # Convert the frame to grayscale as an example preprocessing step
     return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def trigger_decryption(self):
     # Fetch the key entered by the user
     user_key = self.decryption_key_entry.text()
     if not user_key:
        self.show_error_message("Please enter a decryption key.")
        return

     # Use this key for decryption
     self.decrypt_text(user_key)
            
        
    def show_error_message(self, message):
        QMessageBox.critical(self, "Error", message)

    def display_results(self, results):
            for frame_index, extracted_text in results:
                print(f"Frame {frame_index}: Extracted Text: {extracted_text}")
                self.extracted_text_label.append(f"Frame {frame_index}: Extracted Text: {extracted_text}")
                last_text = extracted_text
                
            # Display frame
            self.display_frame(self.video_path, frame_index=0)

            self.update_gui()

    def display_frame(self, video_path, frame_index=0):
     cap = cv2.VideoCapture(video_path)
     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
     ret, frame = cap.read()
     if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
     cap.release()

    def save_text_to_file(self):
            text_to_save = self.extracted_text_label.toPlainText()
            if not text_to_save:
                return

            file_path, _ = QFileDialog.getSaveFileName(None, "Save Text File", "", "Text files (*.txt);;All Files (*)")
            if not file_path:
                return

            with open(file_path, 'w') as file:
                file.write(text_to_save)

            print(f"Text saved to {file_path}")

    def update_gui(self):
            QApplication.processEvents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    text_extractor_window = VideoTextExtractor()
    text_extractor_window.showMaximized()
    sys.exit(app.exec_())



    
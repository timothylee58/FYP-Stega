import cv2
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout,QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,QComboBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np
import random
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad  
from memory_profiler import profile
import base64
import re
import secrets
import reedsolo
import time

start_time = time.time()


class VideoSteganographyApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Steganography using LSB")
        self.setGeometry(100, 100, 500, 450)  # Slightly larger window
        self.setStyleSheet("background-color: #f0f0f0;")
        self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)

        # Font for labels and buttons
        label_font = QFont("Arial", 10, QFont.Bold)
        button_font = QFont("Arial", 10)
        button_style = "QPushButton {background-color: #007bff; color: white; border-radius: 5px;}"
        "QPushButton:hover {background-color: #0056b3;}"
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)  # Space between widgets
        main_layout.setContentsMargins(10, 10, 10, 10)  # Margin around the layout

        # Video file selection
        video_layout = QHBoxLayout()
        self.video_label = QLabel("Select Video File:")
        self.video_label.setFont(label_font)
        self.selected_video_path = QLineEdit()
        self.browse_button = QPushButton("Browse")
        self.browse_button.setFont(button_font)
        self.browse_button.clicked.connect(self.browse_file)
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.selected_video_path)
        video_layout.addWidget(self.browse_button)
        
        #Calculate Total Frames
        self.total_frames_label = QLabel("Total Frames: 0")
        self.total_frames_label.setFont(label_font)
        
        # Add the total frames label to the main layout
        main_layout.addWidget(self.total_frames_label)
        
        # Frame selection
        frame_layout = QHBoxLayout()
        self.frames_label = QLabel("Selected Frames (comma-separated):")
        self.frames_label.setFont(label_font)
        self.selected_frames_entry = QLineEdit()
        frame_layout.addWidget(self.frames_label)
        frame_layout.addWidget(self.selected_frames_entry)

        # Embedded text input
        text_layout = QHBoxLayout()
        self.text_label = QLabel("Enter Embedded Text:")
        self.text_label.setFont(label_font)
        self.embedded_text_entry = QLineEdit()
        text_layout.addWidget(self.text_label)
        text_layout.addWidget(self.embedded_text_entry)

        # Output file path
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Video File:")
        self.output_label.setFont(label_font)
        self.output_video_path = QLineEdit()
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_video_path)

        # Password entry
        password_layout = QHBoxLayout()
        self.password_label = QLabel("Enter Encryption Key/Password:")
        self.password_label.setFont(label_font)
        self.password_entry = QLineEdit()
        self.password_entry.setEchoMode(QLineEdit.Password)
        self.hide_show_btn = QPushButton("Hide/Show")
        self.hide_show_btn.setFont(button_font)
        self.hide_show_btn.clicked.connect(self.toggle_password_visibility)
        password_layout.addWidget(self.password_label)
        password_layout.addWidget(self.password_entry)
        password_layout.addWidget(self.hide_show_btn)
        
        # Encryption Strength Selection
        encryption_strength_layout = QHBoxLayout()
        self.encryption_strength_label = QLabel("Encryption Strength:")
        self.encryption_strength_label.setFont(label_font)
        self.encryption_strength_combo = QComboBox()
        self.encryption_strength_combo.addItems(["AES-128", "AES-256"])
        encryption_strength_layout.addWidget(self.encryption_strength_label)
        encryption_strength_layout.addWidget(self.encryption_strength_combo)

        # Cipher Mode Selection
        cipher_mode_layout = QHBoxLayout()
        self.cipher_mode_label = QLabel("Cipher Mode:")
        self.cipher_mode_label.setFont(label_font)
        self.cipher_mode_combo = QComboBox()
        self.cipher_mode_combo.addItems(["CBC(Cipher Block Chaining)", "GCM(Galois/Counter Mode)"])
        cipher_mode_layout.addWidget(self.cipher_mode_label)
        cipher_mode_layout.addWidget(self.cipher_mode_combo)

        # Add the new layouts to the main layout
        main_layout.addLayout(encryption_strength_layout)
        main_layout.addLayout(cipher_mode_layout)
        
        # Embed button and result label
        embed_layout = QHBoxLayout()
        self.embed_button = QPushButton("Embed Text on Frames")
        self.embed_button.setFont(button_font)
        self.embed_button.clicked.connect(self.embed_text)
        self.result_label = QLabel()
        embed_layout.addWidget(self.embed_button)
        embed_layout.addWidget(self.result_label)
        
       
        # Add all layouts to the main layout
        main_layout.addLayout(video_layout)
        main_layout.addLayout(frame_layout)
        main_layout.addLayout(text_layout)
        main_layout.addLayout(output_layout)
        main_layout.addLayout(password_layout)
        main_layout.addLayout(embed_layout)
        
        self.setLayout(main_layout)

    def browse_file(self):
      file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "All files (*.mp4)")
      if file_path:
        self.selected_video_path.setText(file_path)
        self.calculate_total_frames(file_path)
    
    def toggle_password_visibility(self):
      if self.password_entry.echoMode() == QLineEdit.Password:
            self.password_entry.setEchoMode(QLineEdit.Normal)
      else:
            self.password_entry.setEchoMode(QLineEdit.Password)
        
    def calculate_total_frames(self, video_path):
     cap = cv2.VideoCapture(video_path)
     if not cap.isOpened():
         self.total_frames_label.setText("Total Frames: Unable to open video")
         return 0  # Return 0 if the video cannot be opened
     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
     self.total_frames_label.setText(f"Total Frames: {total_frames}")
     cap.release()
     return total_frames  # Return the total frame count

    @staticmethod
    def get_bit(byte_val, idx):
        return (byte_val >> idx) & 1

    @staticmethod
    def set_bit(byte_val, idx, new_value):
        mask = 1 << idx
        byte_val &= ~mask
        if new_value:
            byte_val |= mask
        return byte_val
    
    def apply_error_correction(self, data):
      rs = reedsolo.RSCodec(10)  # Example: 10 ecc symbols
      data_sequence = list(data)  # Convert generator to list
      return rs.encode(data)
  
    def embed_text(self,encrypted_bytes):
       video_path = self.selected_video_path.text()
       selected_frames = [int(frame.strip()) for frame in self.selected_frames_entry.text().split(",")]
       output_path = self.output_video_path.text()
       embedded_text = self.embedded_text_entry.text()
       password = self.password_entry.text()

       
       # Parse selected frames
       frame_strings = self.selected_frames_entry.text().split(",")
       selected_frames = []
       for frame_str in frame_strings:
        frame_str = frame_str.strip()
        if frame_str.isdigit():  # Check if the string is a number
            selected_frames.append(int(frame_str))
        elif frame_str:  # Non-empty string that is not a number
            self.result_label.setText(f"Error: Invalid frame number '{frame_str}'")
            return

     # Retrieve encryption options from UI
       encryption_strength = self.encryption_strength_combo.currentText()
       cipher_mode = self.cipher_mode_combo.currentText()

     # Determine key length based on user choice
       key_length = 32 if encryption_strength == "AES-256" else 16

     # Encrypt the text using the selected options
       encrypted_text = self.encrypt_text(embedded_text, password, 100000, key_length, cipher_mode)

       try:
        encrypted_bytes = base64.b64decode(encrypted_text)
       except (TypeError, ValueError):
        self.result_label.setText("Error: Failed to decode encrypted text.")
        return

       # Convert encrypted text to bits
       data_bits = list(VideoSteganographyApp.data_to_bits(encrypted_bytes))

       # Ensure encrypted_bytes is a bytes object
       if not isinstance(encrypted_bytes, bytes):
        self.result_label.setText("Error: Encrypted data is not in bytes format.")
        return

        # Apply error correction
       error_corrected_data = self.apply_error_correction(data_bits)
       
       # Embed text on all frames and Pass error-corrected, encrypted data to embed_text_on_all_frames
       total_frames = self.calculate_total_frames(video_path)  # Calculate total frames here
       result = self.embed_text_on_all_frames(video_path, output_path, selected_frames, error_corrected_data, total_frames, text_mappings=None)
       self.result_label.setText(result)

       # Validate base64 string
       if not VideoSteganographyApp.is_valid_base64_string(encrypted_text):
         self.result_label.setText("Error: Encrypted text is not a valid base64 string.")
         return

       # Check data length
       max_data_length = 777600  # Maximum data length for a 1080p frame (6,220,800 bits)
       if len(encrypted_text) * 8 > max_data_length:
         self.result_label.setText("Error: Encrypted text is too long for embedding in the selected frames.")
         return
  
    def __call__(self, msg):
        if "frame" in msg:
            frame_number = int(msg.split()[1])
            progress_percent = int((frame_number / self.n_frames) * 100)
            self.app.result_label.setText(f"Writing video: {progress_percent}%")
            QApplication.processEvents()
            
    def progress_callback(self,current_frame, total_frames):
        progress_percent = int((current_frame / total_frames) * 100)
        self.result_label.setText(f"Writing video: {progress_percent}%")
        QApplication.processEvents()
        
    def write_video_with_progress(self, video_clip, output_filename, fps, codec):
        """ Write the video clip to a file with progress updates. """
        n_frames = int(video_clip.duration * fps)

        video_clip.write_videofile(
            output_filename,
            fps=fps,
            codec=codec,
            write_logfile=False
        )
        
    @staticmethod
    def data_to_bits(data):
     for byte in data:
         for i in range(8):
            yield (byte >> (7 - i)) & 1
    
    @staticmethod
    def is_valid_base64_string(s):
      pattern = r'^[A-Za-z0-9+/]+={0,2}$'
      if re.match(pattern, s):
         return True
      return False
    
    def is_valid_data_length(data, max_length):
     return len(data) <= max_length
        
    def get_existing_text_mappings(self, video_path, selected_frames, text_mappings):
        cap = cv2.VideoCapture(video_path)
        existing_text_mappings = {}
        drawn_text_bboxes = []

        if not cap.isOpened():
            return existing_text_mappings

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index in selected_frames:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_frame)
                font = ImageFont.truetype("arial.ttf", 18)
                text_position = (150, 200)
                text_color = (0, 0, 255)
                text = text_mappings.get(frame_index, "")

                text_bbox = draw.textbbox(text_position, text, font=font)
                if not any(self.bboxes_overlap(text_bbox, existing_bbox) for existing_bbox in drawn_text_bboxes):
                    draw.text(text_position, text, fill=text_color, font=font)
                    drawn_text_bboxes.append(text_bbox)
                    existing_text_mappings[frame_index] = text

            frame_index += 1

        cap.release()
        return existing_text_mappings
        
    def extract_bits_from_pixel(self, pixel, start_bit_position, num_bits):
     extracted_bits = []
     for i in range(num_bits):
        bit = (pixel >> (start_bit_position + i)) & 1
        extracted_bits.append(bit)
     return extracted_bits
 
    def encode_data_into_higher_bits(self, pixel, data_bits, start_bit_position=0):
    
    # Encode each bit from data_bits into the pixel starting from start_bit_position.

     for i, bit in enumerate(data_bits):
        bit_position = start_bit_position + i
        pixel = self.modify_pixel_bit(pixel, bit, bit_position)
     return pixel
 
    def modify_pixel_bit(self, pixel, bit, bit_position):
    
    # Modify a specific bit in the pixel.
    
     mask = 1 << bit_position
     pixel &= ~mask
     if bit:
        pixel |= mask
     return pixel
    
    def modify_pixel(self, frame, data_bits, set_bit):
     bit_gen = iter(data_bits)  # Create an iterator from data_bits
     for y in range(frame.shape[0]):  # Iterate over rows
        for x in range(frame.shape[1]):  # Iterate over columns
             for i in range(3):  # Assuming RGB
                try:
                    # Get next bit and modify the pixel
                    bit = next(bit_gen)
                    frame[y, x, i] = set_bit(frame[y, x, i], 2, bit)  # Example: modify the 3rd least significant bit
                except StopIteration:
                    # If no more bits, stop the modification
                    return frame
     return frame

    def bboxes_overlap(self, bbox1, bbox2):
      left1, upper1, right1, lower1 = bbox1
      left2, upper2, right2, lower2 = bbox2

    # Check for overlap
      if right1 < left2 or left1 > right2:
        return False
      if lower1 < upper2 or upper1 > lower2:
        return False

      return True

    def encrypt_text(self, text, password, iterations, key_length=32, mode='CBC'):
     mode = mode.split('(')[0] 
     
     
     # Validate input parameters
     if not text:
        raise ValueError("Input text cannot be empty")

     # Key derivation based on the chosen key length
     salt = secrets.token_bytes(16)
     kdf = PBKDF2HMAC(
     algorithm=hashes.SHA256(),
     length=key_length,
     salt=salt,
     iterations=iterations,
     backend=default_backend()
)
     key = kdf.derive(password.encode())

     # Cipher mode selection
     iv = secrets.token_bytes(16)
     if mode == 'CBC':
        cipher = AES.new(key, AES.MODE_CBC, iv)
     elif mode == 'GCM':
        cipher = AES.new(key, AES.MODE_GCM, iv)
     else:
        raise ValueError(f"Unsupported cipher mode: {mode}")
     
     padded_text = self.pad_text(text)
     encrypted_text = cipher.encrypt(padded_text)
     base64_encoded = base64.b64encode(iv + salt + encrypted_text)
     encoded_string = base64_encoded.decode('utf-8')  # Decode to string
    
     # Debugging: Log the lengths of the components
     print(f"IV Length: {len(iv)}, Salt Length: {len(salt)}, Encrypted Text Length: {len(encrypted_text)}")
     print(f"Base64 Encoded String Length: {len(encoded_string)}")

     return encoded_string
     
    def pad_text(self, text):
        block_size = AES.block_size
        padded_text = pad(text.encode(), block_size)
        return padded_text
    
    @profile
    def embed_text_on_all_frames(self, video_path, output_path, selected_frames, error_corrected_data, total_frames, text_mappings=None):
     cap = cv2.VideoCapture(video_path)
     if not cap.isOpened():
         return "Error: Could not open video file."
    
     # Video writer setup
     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
     frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
     fourcc = cv2.VideoWriter_fourcc(*'H264')  
     output_path += '.mp4' if not output_path.lower().endswith('.mp4') else ''
     out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))
     
     frame_index = 0
     bit_gen = iter(error_corrected_data)
     embedded_text = ""

     while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index in selected_frames:
            try:
                frame = self.modify_pixel(frame, bit_gen, self.set_bit)  # Embed the data bits into the frame
                
                # Convert frame to PIL Image for drawing text
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_frame)
                font = ImageFont.truetype("arial.ttf", size=20)  # Larger font size
                text_position = (50, 50)  # Adjust position as needed
                text_color = self.get_contrast_color(frame, text_position)  # Function to get contrasting color
                
                # Use the text provided by the user
                user_text = self.embedded_text_entry.text()
                draw.text(text_position, user_text, font=font, fill=text_color)  # Draw user-provided text
                
                frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)  

            except Exception as e:
                print(f"An error occurred during frame processing: {e}")
                continue

        try:
            out.write(frame)  # Write every frame, modified or not
        except Exception as e:
            print(f"An error occurred while writing the frame: {e}")

        frame_index += 1  # Increment frame index

        # Update progress
        progress = f"Progress: {frame_index}/{total_frames} frames processed"
        self.result_label.setText(progress)
        QApplication.processEvents()

     cap.release()
     out.release()
     
      # Combine audio with the modified video
     modified_video = VideoFileClip(output_path)
     original_audio = VideoFileClip(video_path).audio
     modified_video = modified_video.set_audio(original_audio)

     # Save the final video with audio
     final_output_path = "final_" + output_path
     modified_video.write_videofile(final_output_path, codec='libx264')
     original_audio = VideoFileClip(video_path).audio
     modified_video = modified_video.set_audio(original_audio)

     # Use custom video writing function
     self.write_video_with_progress(modified_video, final_output_path, fps=modified_video.fps, codec='libx264')
 
     if frame_index == 0:
        return "Error: No frames were processed."
     else:
        return f"Text '{embedded_text}' embedded in selected frames. Output video saved as {output_path}"

    def get_contrast_color(self, frame, position):
       
        pixel = frame[position[1], position[0]]  # Get the pixel color at the position
        if np.mean(pixel) > 128:
            return (0, 0, 0)  # Dark text for light backgrounds
        else:
            return (255, 255, 255)  # Light text for dark backgrounds

    def get_total_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames
    
class CustomLogger:
    def __init__(self, app, n_frames):
        self.app = app
        self.n_frames = n_frames

if __name__ == "__main__":
    start_time = time.time()  # Start time measurement

    app = QApplication([])
    video_text_embedder_app = VideoSteganographyApp()
    video_text_embedder_app.showMaximized()
    app.exec_()

    end_time = time.time()  # End time measurement
    processing_time = end_time - start_time
    print(f"Processing Time: {processing_time} seconds")

#Note
#For a 1080p video frame (1920x1080 pixels), and assuming you're using all three color channels (RGB) for embedding,
#the maximum amount of data you could theoretically embed in a single frame would be:1920 pixels * 1080 pixels * 3 color channels = 6,220,800 bits
#Dividing by 8 gives you the number of bytes:
#6,220,800 bits / 8 = 777,600 bytes

#Note
#The OpenCV library uses BGR format to represent colors. In this format, each channel is represented using 8-bits. 
#Bytes per Frame = (1920 * 1080 * 2) / 8 = 518,400 bytes
#Total Embedding Capacity = 518,400 bytes * 100 frames = 51,840,000 bytes (or approximately 49.5 MB)
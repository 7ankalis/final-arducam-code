import serial
import cv2
import numpy as np
import time
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Function to detect and encode faces
def detect_and_encode(image):
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            faces = []
            for box in boxes:
                face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0)
                encoding = resnet(face_tensor).detach().numpy().flatten()
                faces.append(encoding)
            return faces
    return []

# Function to encode specific images with predefined names
def encode_known_faces(known_faces):
    known_face_encodings = []
    known_face_names = []

    for name, image_path in known_faces.items():
        known_image = cv2.imread(image_path)
        if known_image is not None:
            known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
            encodings = detect_and_encode(known_image_rgb)
            if encodings:
                known_face_encodings.append(encodings[0])  # Assuming one face per image
                known_face_names.append(name)

    return known_face_encodings, known_face_names

# Define known faces with explicit names
known_faces = {
    "name1": "images/name1.jpg",
    "name2": "images/name2.jpg",
    "name3": "images/name3.jpg",
    "name4": "images/name4.jpg",
    "name5": "images/name5.jpg",
    "name6": "images/name6.jpg",
}

# Encode known faces
known_face_encodings, known_face_names = encode_known_faces(known_faces)

# Function to recognize faces
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized_names = []
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < threshold:
            recognized_names.append(known_names[min_distance_idx])
        else:
            recognized_names.append('Not Recognized')
    return recognized_names

# Serial port configuration
SERIAL_PORT = 'COM3'  # Replace with your Arduino's serial port
BAUD_RATE = 115200    # Match this with the Arduino's baud rate

# Commands
START_STREAM = bytes([0x20])  # Command to start video streaming
STOP_STREAM = bytes([0x21])   # Command to stop video streaming

def receive_frame(ser):
    """
    Receive a single frame from the Arduino.
    """
    frame_data = b""
    start_marker = ser.read(2)  # Read the start marker (0xFF, 0xAA)
    if start_marker == b'\xff\xaa':
        while True:
            chunk = ser.read(ser.in_waiting or 1)  # Read available data
            if not chunk:
                break
            frame_data += chunk
            if frame_data[-2:] == b'\xbb\xcc':  # Check for the end marker
                break
    return frame_data[:-2]  # Remove the end marker (0xBB, 0xCC)

def main():
    # Initialize serial connection
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for the connection to establish
        print(f"Connected to Arduino on {SERIAL_PORT} at {BAUD_RATE} baud.")
    except Exception as e:
        print(f"Error opening serial port: {e}")
        return

    try:
        # Start video streaming
        ser.write(START_STREAM)
        print("Video streaming started. Press 'q' to stop.")

        while True:
            # Receive a frame
            frame_data = receive_frame(ser)
            if frame_data:
                # Decode the JPEG frame
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    # Convert frame to RGB for face recognition
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detect and encode faces in the frame
                    test_face_encodings = detect_and_encode(frame_rgb)

                    # Recognize faces if any are detected
                    if test_face_encodings and known_face_encodings:
                        names = recognize_faces(np.array(known_face_encodings), known_face_names, test_face_encodings)
                        boxes, _ = mtcnn.detect(frame_rgb)
                        if boxes is not None:
                            for name, box in zip(names, boxes):
                                (x1, y1, x2, y2) = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    # Display the frame
                    cv2.imshow("ArduCAM Video Stream with Face Recognition", frame)

            # Stop streaming if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                ser.write(STOP_STREAM)
                break

    finally:
        ser.close()
        cv2.destroyAllWindows()
        print("Video streaming stopped. Serial connection closed.")

if __name__ == "__main__":
    main()

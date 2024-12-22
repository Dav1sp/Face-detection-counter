import cv2
import face_recognition
import time
import numpy as np

def track_unique_faces(video_source=0, detection_duration=10):
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    known_face_encodings = []
    unique_faces_count = 0
    start_time = time.time()

    print("Starting face tracking...")

    while time.time() - start_time < detection_duration:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Convert the frame to RGB (face_recognition uses RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and encode them
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            if not any(matches):
                # Add the new face encoding to the known faces list
                known_face_encodings.append(face_encoding)
                unique_faces_count += 1

            # Draw a rectangle around the face (optional)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the frame (optional)
        cv2.imshow("Face Tracking", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Number of unique faces detected: {unique_faces_count}")
    with open('output.txt', "w") as file:
        file.write(f"{unique_faces_count}")
    return unique_faces_count

# Example usage
track_unique_faces(video_source=0, detection_duration=10)

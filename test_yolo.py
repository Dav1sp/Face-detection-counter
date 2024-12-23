from ultralytics import YOLO
import cv2
import time
'''
rpicam-hello -t 0s --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json --viewfinder-width 1920 --viewfinder-height 1080 --framerate 30

'''
def track_unique_faces(video_source=0, detection_duration=10, output_file="unique_faces_count.txt"):
    # Load YOLOv8 model (you can use a custom model trained for faces)
    model = YOLO("yolov11n-face.pt")  # Use "yolov8n-face.pt" if you have a face-specific model
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    unique_faces = []  # Store bounding boxes of detected faces
    forgotten_faces = []  # Faces that have been forgotten
    forgotten_timeout = detection_duration  # Time (seconds) to forget a face
    unique_faces_count = 0
    start_time = time.time()

    print("Starting face tracking...")

    while time.time() - start_time < detection_duration:

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        current_time = time.time()

        # Detect faces with YOLOv8
        results = model(frame)
        faces = []
        for result in results:
            for box in result.boxes.xyxy:  # Bounding boxes (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box)
                faces.append((x1, y1, x2 - x1, y2 - y1))

        # Check if faces are new
        for (x, y, w, h) in faces:
            new_face = True
            for (prev_x, prev_y, prev_w, prev_h, last_seen) in unique_faces:
                # Check overlap using IoU or proximity
                if (x < prev_x + prev_w and x + w > prev_x and y < prev_y + prev_h and y + h > prev_y):
                    unique_faces.remove((prev_x, prev_y, prev_w, prev_h, last_seen))
                    unique_faces.append((x, y, w, h, current_time))
                    new_face = False
                    break

            # If face is new and not in forgotten list
            if new_face:
                is_forgotten = False
                for (old_x, old_y, old_w, old_h, forgotten_time) in forgotten_faces:
                    if (x < old_x + old_w and x + w > old_x and y < old_y + old_h and y + h > old_y):
                        if current_time - forgotten_time > forgotten_timeout:
                            forgotten_faces.remove((old_x, old_y, old_w, old_h, forgotten_time))
                        else:
                            is_forgotten = True
                            break

                if not is_forgotten:
                    unique_faces.append((x, y, w, h, current_time))
                    unique_faces_count += 1

        # Forget faces that have been out of view for a while
        unique_faces = [(x, y, w, h, last_seen) for (x, y, w, h, last_seen) in unique_faces if current_time - last_seen < forgotten_timeout]

        # Update forgotten faces
        for (prev_x, prev_y, prev_w, prev_h, last_seen) in unique_faces:
            if current_time - last_seen >= forgotten_timeout:
                forgotten_faces.append((prev_x, prev_y, prev_w, prev_h, last_seen))

        # Draw rectangles around detected faces
        for (x, y, w, h, _) in unique_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Face Tracking", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save the unique face count
    with open(output_file, "w") as file:
        file.write(f"Number of unique faces detected: {unique_faces_count}\n")

    print(f"Number of unique faces detected: {unique_faces_count}")

    return unique_faces_count

# Example usage
track_unique_faces(video_source=0, detection_duration=10)

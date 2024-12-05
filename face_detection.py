import cv2
import time

def track_unique_faces(video_source=0, detection_duration=10):
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    trackers = []
    unique_faces_count = 0
    start_time = time.time()
    
    print("Starting face tracking...")
    
    while time.time() - start_time < detection_duration:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Convert frame to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the current frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        new_trackers = []
        for tracker in trackers:
            success, bbox = tracker.update(frame)
            if success:
                new_trackers.append(tracker)
                x, y, w, h = map(int, bbox)
                # Draw rectangle for tracked faces (optional)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        trackers = new_trackers

        # Update trackers and add new ones
        for (x, y, w, h) in faces:
            overlap = False
            for tracker in trackers:
                # Check if the detected face overlaps with an existing tracker
                success, bbox = tracker.update(frame)
                if not success:
                    continue
                tx, ty, tw, th = map(int, bbox)
                if (x < tx + tw and x + w > tx and y < ty + th and y + h > ty):
                    overlap = True
                    break

            if not overlap:
                # Create a new tracker for this face
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (x, y, w, h))
                trackers.append(tracker)
                unique_faces_count += 1

        # Display the frame (optional)
        cv2.imshow("Face Tracking", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Number of unique faces detected: {unique_faces_count}")
    return unique_faces_count

# Example usage
track_unique_faces(video_source=0, detection_duration=5)
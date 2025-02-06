import cv2
import time
import json
import os
import numpy as np
import torch
from ultralytics import YOLO
from picamera2 import Picamera2
from PIL import Image
from facenet_pytorch import InceptionResnetV1

def preprocess_face(face_image):
    # Converte a imagem recortada da face para RGB e redimensiona para 160x160
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    face_pil = face_pil.resize((160, 160))  # O InceptionResnet requer 160x160
    face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float() / 255.0  # Normaliza entre [0, 1]
    face_tensor = face_tensor.unsqueeze(0)  
    return face_tensor


def get_face_embedding(frame, face_coordinates, inception_resnet):
    face = frame[face_coordinates[1]:face_coordinates[3], face_coordinates[0]:face_coordinates[2]]
    face_tensor = preprocess_face(face)

    face_tensor = face_tensor.to(device='cpu')  # Ou 'cuda' se usar GPU
    
    # Calcula o embedding usando o InceptionResnet
    with torch.no_grad():
        embedding = inception_resnet(face_tensor)
    return embedding

def save(unique_faces_count, unique_faces, frame,time,output_dir="cache",max_cache_size=5):
    data = {
        'time':time,
        'unique_faces_count': unique_faces_count,
        'unique_faces': [face[4].tolist() for face in unique_faces],
    }
    print(unique_faces)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = os.listdir(output_dir)
    indices = []
    
    for file in files:
        if file.startswith("frame_"):
            idx = int(file.split("_")[1].split(".")[0])
            indices.append(idx)

    next_index = 1 if not indices else max(indices) + 1

    if len(indices) >= max_cache_size:
        oldest_index = min(indices)
        
        os.remove(os.path.join(output_dir, f"frame_{oldest_index}.jpg"))
        os.remove(os.path.join(output_dir, f"unique_faces_count_{oldest_index}.json"))

    cv2.imwrite(os.path.join(output_dir, f"frame_{next_index}.jpg"), frame)
    with open(os.path.join(output_dir, f"unique_faces_count_{next_index}.json"), 'w') as json_out:
        json.dump(data, json_out, indent=4)


def track_unique_faces(video_source=0):
    
    model = YOLO("yolov11n-face.pt")  # pode usar "yolov8n-face.pt"
    inception_resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Inicializa a captura de video
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration()  # Get the configuration
    picam2.configure(preview_config)
    picam2.start()
    
    while True:
        unique_faces = [] #Guarda as caras das pessoas
        unique_faces_count = 0

        print("Starting face tracking...")

        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        current_time = time.time()

        results = model(frame)
        faces = []
        for result in results:
            for box in result.boxes.xyxy:  # Bounding boxes (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box)
                faces.append((x1, y1, x2 - x1, y2 - y1))

        # Check the faces
        for (x, y, w, h) in faces:
            embedding = get_face_embedding(frame, [x,y,w+x,h+y], inception_resnet)
            unique_faces.append((x,y,w,h,embedding))
            unique_faces_count +=1    
            
        # Draw rectangles and the blur around detected faces
        for (x, y, w, h, _) in unique_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_region = frame[y:y+h,x:x+w]
            blurred_face = cv2.GaussianBlur(face_region, (51,51),30)
            frame[y:y+h,x:x+w] = blurred_face

        # Display the frame
        #cv2.imshow("Face Tracking", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        save(unique_faces_count,unique_faces,frame,current_time)
        
        time.sleep(10)

# Example usage
track_unique_faces(video_source="/dev/video0")

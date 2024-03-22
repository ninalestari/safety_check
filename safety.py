import cv2
from cv2 import imshow
from ultralytics import YOLO
import face_recognition
import numpy as np
import os
import pymongo
from PIL import Image, ImageDraw#, UnidentifiedImageError
from db_operations import insert_document
from datetime import datetime
import time
import subprocess

# Melatih model
command = 'yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=224 plots=True'

result = subprocess.run(command, shell=True, capture_output=True, text=True)

# Mencetak output dan error jika ada
print("Output:", result.stdout)
if result.stderr:
    print("Error:", result.stderr)


def save(data):
    # print("Hello, " + data + "!")
    mongo_uri = "mongodb://localhost:27017/"
    database_name = "safety"
    collection_name = "results"
    data_to_insert = data

    # Calling the insert_document function
    inserted_id = insert_document(mongo_uri, database_name, collection_name, data_to_insert)

    print("Inserted document ID:", inserted_id)

# Load the YOLOv8 model
model = YOLO('best.pt')

stream_url1 = 'http://172.110.2.204'
print ("ESP32-CAM Video Stream URL: ", stream_url1)
cap = cv2.VideoCapture(stream_url1+':81/stream')
#cap = cv2.VideoCapture(0)
check1 = cap.isOpened()
print ("ESP32-CAM1 Video Stream Status: ", check1)


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        for result in results:
            boxes = result.boxes.cpu().numpy()  # Get boxes on CPU in numpy format
            for box in boxes:  # Iterate over boxes
                r = box.xyxy[0].astype(int)  # Get corner points as int
                class_id = int(box.cls[0])  # Get class ID
                class_name = model.names[class_id]  # Get class name using the class ID
                print(f"Detected: {class_name}")#, Box: {r}")  # Print class name and box coordinates
                cv2.rectangle(frame, r[:2], r[2:], (0, 255, 0), 2,cv2.FILLED)  # Draw boxes on the image
                # Draw a label with a name below the face
                
                font = cv2.FONT_HERSHEY_DUPLEX
                
                current_time = time.time()
                dt = datetime.fromtimestamp(current_time)
                fdt2 = dt.strftime("%d-%m-%Y %H:%M:%S")
                ts = dt.strftime("%d%m%Y%H%M%S")
                print(f"Face detected:", class_name, 'time:',fdt2)
                
                pil_image = Image.fromarray(frame)
                # Create a Pillow ImageDraw Draw instance to draw with
                draw = ImageDraw.Draw(pil_image)

                pil_image.save("D:\computer_vision\identified\{}-{}.jpg".format(class_name,ts)) #menyimpan di FTP
                #creds = get_credentials()
                #service = build('drive', 'v3', credentials=creds)
                #folder_id = find_or_create_folder(service, 'hasil')
                #upload_file(service, '{}-{}.jpg'.format(class_name,ts), 'D:\computer_vision\identified/{}-{}.jpg'.format(class_name,ts), 'image/jpeg', folder_id)
                # time.sleep(30)
                #print("New Face Model saved:", name)
                access = False
                # class_name = "no safety"
                if class_name == "no safety":
                    access = False
                else:
                    access = True


                data_to_save = {
                        "guid":"da7d8c57-6d1c-4143-b43c-2a183c282702",
                        "name": class_name,
                        "id": "12345678",
                        "position": "Teknisi",
                        "image": "{}-{}.jpg".format(class_name, ts),
                        "access": access,
                        "status":class_name,
                        "date": fdt2,
                       
                            
                    }
                    

                print(data_to_save)
                save(data_to_save)
        

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8", annotated_frame)
        print(f'Detection frame', results)    

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import os
import time
import chrysalis
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import timezone
import datetime
# import module sys to get the type of exception
import sys
import psutil

# issue for RTX GPU: https://github.com/tensorflow/tensorflow/issues/24496
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

os.environ['DISPLAY'] = ":0"

default_confidence = 0.5

# Fetch the service account key JSON file contents
cred = credentials.Certificate('utplcovid-firebase-adminsdk-ytv22-cd40c21acc.json')
# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred,  {
    'databaseURL': 'https://utplcovid-default-rtdb.firebaseio.com/'
})

def detect_mask(img, face_detector, mask_detector, confidence_threshold, image_show=False):
    # Initialize the labels and colors for bounding boxes
    num_class = mask_detector.layers[-1].get_output_at(0).get_shape()[-1]
    labels, colors = None, None
    if num_class == 3:
        labels = ["Face with Mask Incorrect", "Face with Mask Correct", "Face without Mask"]
        colors = [(0, 255, 255), (0, 255, 0), (0, 0, 255)]
    elif num_class == 2:
        labels = ["Face with Mask", "Face without Mask"]
        colors = [(0, 255, 0), (0, 0, 255)]

    # Load the input image from disk, clone it, and grab the image spatial dimensions
    (h, w) = img.shape[:2]

    # Construct a blob from the image
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    face_detector.setInput(blob)
    detections = face_detector.forward()

    # Record status
    # MFN: 0 is "mask correctly" and 1 is "no mask"
    # RMFD: 0 is "mask correctly", 1 is "mask incorrectly", and 2 is "no mask"
    status = 0
    sceneExtra = 40

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > confidence_threshold:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

            # Extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
            face = img[start_y:end_y, start_x:end_x]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Pass the face through the model to determine if the face has a mask or not
            prediction = mask_detector.predict(face)[0]
            label_idx = np.argmax(prediction)

            # Determine the class label and color we'll use to draw the bounding box and text
            label = labels[label_idx]
            color = colors[label_idx]

            # Update the status
            if num_class == 3:
                if label_idx == 0:
                    temp = 1
                elif label_idx == 1:
                    temp = 0
                else:
                    temp = 2
                status = max(status, temp)
            elif num_class == 2:
                status = max(status, label_idx)

            # Include the probability in the label
            label = "{}: {:.2f}%".format(label, max(prediction) * 100)

            # Display the label and bounding box rectangle on the output frame
            cv2.putText(img, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), color, 2)
            print("[INFO] Status detection: " + str(status) + " - label: " + label)

            if status > 0:
                try:
                    print("[INFO] guardando imagen...", datetime.datetime.now().astimezone().isoformat())
                    sub_face = img[start_y-sceneExtra:end_y+sceneExtra, start_x-sceneExtra:end_x+sceneExtra]
                    imgName = 'detections/Frame-'+ time.strftime("%Y_%m_%d_%H_%M_%S")+ '.jpg'
                    cv2.imwrite(imgName, sub_face)
                    ref = db.reference('facemask')
                    virtualM = psutil.virtual_memory()
                    virtualMemoryT = virtualM.total >> 30
                    ref.push({
                        'fecha': datetime.datetime.now().astimezone().isoformat(),
                        'cpuP': psutil.cpu_percent(interval=1),
                        'virtualMemoryT': virtualMemoryT,
                        'virtualMemoryP': virtualM.percent,
                        'virtualMemory': virtualM.used >> 30,
                        'imagen': imgName,
                        'nodo': 'CLOUD_DECODER'
                    })
                except Exception as e:
                    print("[INFO] Error al guardar la imagen...", sys.exc_info()[0])
                    print(e)
                    break

        else:
            break

    return status, img


def process_captured_video(camera, faceDetector, maskDetector, confidenceThreshold):
    
    c = 1
    frameRate = 15 # Frame number interception interval (one frame is intercepted every 100 frames)

    while True:
        success, frame = camera.read()
        if not success:
            print("[ERROR] Camera is disconnected...")
            camera.release()
            return False
        else:
            if(c % frameRate == 0):
                print("Start to capture video:" + str(c) + "frame")
                # Detect faces in the frame and determine if they are wearing a face mask or not
                detect_mask(frame, faceDetector, maskDetector, confidenceThreshold)
                cv2.imshow('Frame', frame)
            c += 1  
            
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return True

if __name__ == "__main__":

    print("[INFO] loading face detector model...")
    prototxtPath ="./face_detector_model/deploy.prototxt"
    weightsPath = "./face_detector_model/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


    print("[INFO] loading face mask detector model...")
    maskNet = load_model("./mask_detector_models/mask_detector_MFN.h5")
    
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")

    while True:
        print("[INFO] Tratando de conectar...")
        capture = cv2.VideoCapture("rtsp://159.69.217.242:9665/mystream")
        if capture.isOpened():
            print("[INFO] Conexion correcta a camara...")
            response = process_captured_video(capture, faceNet, maskNet, 0.7)
            if not response :
                time.sleep(10)
                continue
            
        else:
            time.sleep(3)
            print("[ERROR] Error al conectar a camara...")
            capture.release()
            cv2.destroyAllWindows()
            continue
    
    # Loop over the frames from the video stream
    while True:
        # Grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
        success, frame = capture.read()

        if  success:
            # Show the output frame
            cv2.imshow("Frame", frame)
        else:
            print("[ERROR] Tratando de reconectar...")
            time.sleep(3)
        
        # Detect faces in the frame and determine if they are wearing a face mask or not
        #detect_mask(frame, face_detector, mask_detector, confidence_threshold)

       
        key = cv2.waitKey(1) & 0xFF

        # If the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()
    i = 0
    '''
    while True:
        img = chrys.VideoLatestImage()
        if img is not None:
            frame = img.data
            frame = imutils.resize(frame, width=400)
            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            # loop over the detected face locations and their corresponding locations
            for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output frame
                cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
                if mask < withoutMask:
                   print("[INFO] guardando imagen...")
                   imgName = 'detections/Frame'+str(i)+'.jpg'
                   cv2.imwrite(imgName, frame)
                   
                   # Getting the current date
                   # and time
                   dt = datetime.datetime.now(timezone.utc)
                      
                   utc_time = dt.replace(tzinfo=timezone.utc)
                   utc_timestamp = utc_time.timestamp()
                   
                   ref = db.reference('facemask')
                   ref.push({
                        'fecha': utc_timestamp,
                        'cpu': 7,
                        'imagen': imgName
                   })
                   i += 1
                
            # show the output frame
            cv2.imshow("mask_detector", frame)
            key = cv2.waitKey(1) & 0xFF
	
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
    '''
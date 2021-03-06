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

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > default_confidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

def validate_param(param_name):
    """
    Validating OS environment variables

    """
    if param_name not in os.environ:
        raise ValueError("missing environment variable " + param_name)
    return os.environ[param_name]

if __name__ == "__main__":

    port = '6113'
    host = 'casabelen.monstrous-buildsergeant-3oljx.chrysvideo.com'
    password = 'bbcb9c9cd13a765ad6b83f996884eb3c'
    cert_path = 'chryscloud.cer'

    chrys = chrysalis.Connect(host=host, port=port, password=password, ssl_ca_cert=cert_path)

    print("[INFO] loading face detector model...")
    prototxtPath ="deploy.prototxt"
    weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    print("[INFO] loading face mask detector model...")
    maskNet = load_model("mask_detector.model")
    
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    i = 0
    while True:
        time.sleep(1) 
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
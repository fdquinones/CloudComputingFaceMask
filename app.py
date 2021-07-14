import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response
import numpy as np
import imutils
import cv2
import os
import time
import chrysalis
import imutils

# issue for RTX GPU: https://github.com/tensorflow/tensorflow/issues/24496
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

os.environ['DISPLAY'] = ":0"

default_confidence = 0.5

app = Flask(__name__)

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
    print("[INFO] PARAM_NAME:", param_name, "PARAM_VALUE: ", os.environ[param_name])
    return os.environ[param_name]

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        img = chrys.VideoLatestImage()
        if img is  None:
            break
        else
            frame = img.data
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
                   
                   
@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

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
    
    # arrancar servidor web
    app.run(host='0.0.0.0', debug=True)
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")

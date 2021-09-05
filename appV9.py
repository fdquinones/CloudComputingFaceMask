# import required libraries
import cv2
import time
import sys
import os
import numpy as np
from imutils.video import FPS
from camgear import CamGearUtpl
from firebase_admin import credentials
from firebase_admin import db
import firebase_admin
import psutil
import datetime
import asyncio
import paho.mqtt.client as mqtt
import json
import csv

async def publishDatabase(refDatabase: any, labelMask:str, prediction:int, imgName:str) -> None:
    print("[INFO] publicando resultados...")
    virtualM = psutil.virtual_memory()
    refDatabase.push({
                        'fecha': datetime.datetime.now().astimezone().isoformat(),
                        'cpuP': psutil.cpu_percent(interval=1),
                        'virtualMemoryT': virtualM.total >> 30,
                        'virtualMemoryP': virtualM.percent,
                        'virtualMemory': virtualM.used >> 30,
                        'status': 1,
                        'label': labelMask,
                        'prediction': prediction,
                        'imagen': imgName,
                        'nodo': 'CLOUD_DECODER'
                    })
def publishMqtt(clientMqtt: any, labelMask:str, prediction:int, imgName:str) -> None:
    print("[INFO] publicando resultados a mqtt...")
    virtualM = psutil.virtual_memory()
    clientMqtt.publish ('cloud/camera', json.dumps({
                        'fecha': datetime.datetime.now().astimezone().isoformat(),
                        'cpuP': psutil.cpu_percent(),
                        'virtualMemoryT': virtualM.total >> 30,
                        'virtualMemoryP': virtualM.percent,
                        'virtualMemory': virtualM.used >> 30,
                        'status': 1,
                        'label': labelMask,
                        'prediction': prediction,
                        'imagen': imgName,
                        'nodo': 'CLOUD_DECODER'
                    }))

def publishCsv() -> None:
    with open('detections/metrics.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SN", "Movie", "Protagonist"])
        writer.writerow([1, "Lord of the Rings", "Frodo Baggins"])
        writer.writerow([2, "Harry Potter", "Harry Potter"])

def publishHeaderCsv() -> None:
    with open('detections/metrics.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["CPU%", "RAM Total (GB)", "RAM Usada (GB)", "RAM (%)", "Etiqueta", "Prediccion(%)", "Imagen"])

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("$SYS/#")

async def main():
    #publishHeaderCsv()
    #cargando conexion con mqtt
    #clientMqtt = mqtt.Client(client_id="nodedetector")
    #clientMqtt.on_connect = on_connect

    #clientMqtt.connect("mqtt.eclipseprojects.io", 1883, 60)
    
    
    #Configuracion del modelo
    #INPUT_FILE = os.getenv('INPUTFILE', 'rtsp://192.168.0.100:5540/ch0')
    SHOW_IMAGE = os.getenv('SHOWIMAGE', True)
    #INPUT_FILE='rtsp://192.168.0.100:5540/ch0'
    #INPUT_FILE='rtsp://fdquinones:1104575012@190.96.102.151:15540/stream2'
    #INPUT_FILE='rtsp://fdquinones:1104575012@192.168.1.100:554/stream2'
    INPUT_FILE='D://grabaciones//20201217_221605_tp00014.mp4'
    OUTPUT_FILE='output.avi'
    LABELS_FILE='yolo_model/obj.names'
    CONFIG_FILE='yolo_model/yolov3_tiny_ygb.cfg'
    WEIGHTS_FILE='yolo_model/yolov3_tiny_ygb_last.weights'
    CONFIDENCE_THRESHOLD=0.3

    H=None
    W=None

    #leer el archivo de etiquetas
    LABELS = open(LABELS_FILE).read().strip().split("\n")
    print(LABELS)

    #generar colores para cada etiqueta
    COLORS = [(0, 255, 0), (0, 0, 255), (0, 255, 255)]
    print(COLORS)

    #cargar modelo  yolo
    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)


    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    print(ln)
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print(ln)

    # formatting parameters as dictionary attributes
    #options = {"CAP_PROP_FRAME_WIDTH":416, "CAP_PROP_FRAME_HEIGHT":416, "CAP_PROP_FPS":15}
    options = {"CAP_PROP_FPS":30}


    #INICIALIZAR LA BASE DE DATOS
    cred = credentials.Certificate('utplcovid-firebase-adminsdk-ytv22-cd40c21acc.json')
    # Initialize the app with a service account, granting admin privileges
    firebase_admin.initialize_app(cred,  {
        'databaseURL': 'https://utplcovid-default-rtdb.firebaseio.com/'
    })

    # initialize database
    refDatabase = db.reference('facemask')
    print("[INFO] starting reference database")

    #Defincion de frame para procesar
    c = 1
    frameRate = 30 # Frame number interception interval (one frame is intercepted every 100 frames)

    # Open suitable video stream, such as webcam on first index(i.e. 0)
    while True:
        try:
            print("[INFO] Estableciendo conexion con la camara {}".format(INPUT_FILE))
            stream = CamGearUtpl(source=INPUT_FILE, logging=True, **options)
            break #  Se rompe la espera de conexion de la camara
        except RuntimeError as e:
            print("[INFO] Error  al leer camara remota, se vuelve a intentar...", sys.exc_info()[0])
            print(e)
            time.sleep(5)

    #Inicia el flujo de lectura
    stream.start()

    # start the FPS timer
    fps = FPS().start()

    isOpen = stream.stream.isOpened()
    print("is open:{}".format(isOpen))

    # loop over
    while True:

        # read frames from stream
        frame = stream.read()

        # check for frame if Nonetype
        if frame is None:
            break
        
        if(c % frameRate == 0):
            print("ingreso a procesar")
            # {do something with the frame here}
            time_time = time.time()
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            layerOutputs = net.forward(ln)
            #print(layerOutputs)
            print("cost time:{}".format(time.time() - time_time))
                
            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > CONFIDENCE_THRESHOLD:
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD)

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    color = [int(c) for c in COLORS[classIDs[i]]]

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    labelMask = LABELS[classIDs[i]]
                    prediction = confidences[i]
                    text = "{}: {:.4f}".format(labelMask, prediction)
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

                    try:
                        print("[INFO] guardando imagen...")
                        imgName = 'detections/Frame-'+ time.strftime("%Y_%m_%d_%H_%M_%S") + '_' + str(c) + '.jpg'
                        cv2.imwrite(imgName, frame)

                        timeTimePublish = time.time()
                        

                        #publishDatabase(refDatabase=refDatabase, labelMask = labelMask, prediction = prediction, imgName=imgName )

                        #publishMqtt(clientMqtt=clientMqtt, labelMask = labelMask, prediction = prediction, imgName=imgName )
                        
                        print("cost time publish:{}".format(time.time() - timeTimePublish))
                    except Exception as e:
                        print("[INFO] Error al guardar la imagen...", sys.exc_info()[0])
                        print(e)
                        continue    
        c += 1
        fps.update()
        
        # Show output window, siempre que este activado
        if SHOW_IMAGE:
            cv2.imshow("Output", frame)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # close output window
    cv2.destroyAllWindows()

    # safely close video stream
    stream.stop()

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


asyncio.run(main())
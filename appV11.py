# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import sys
import os
import csv
from firebase_admin import credentials
from firebase_admin import db
import firebase_admin
import psutil
import datetime

INPUT_FILE = 'rtsp://159.69.217.242:9665/mystream'
SHOW_IMAGE = os.getenv('SHOW_IMAGE', True)

print("------------------------imprimiendo la version de cv2----------------")
print(cv2.__version__)

def publishDatabase(refDatabase: any, labelMask:str, prediction:int, imgName:str) -> None:
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
def publishCsv(labelMask:str, prediction:int, imgName:str) -> None:
    print("[INFO] publicando resultados a csv...")
    virtualM = psutil.virtual_memory()
    with open('detections/metrics.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.datetime.now().astimezone().isoformat()
        , psutil.cpu_percent()
        ,virtualM.total >> 30
        ,virtualM.percent
        ,virtualM.used >> 30
        ,1
        ,labelMask
        ,prediction
        ,imgName])

def publishHeaderCsv() -> None:
    with open('detections/metrics.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Fecha", "CPU%", "RAM Total (GB)", "RAM Usada (GB)", "RAM (%)", "Etiqueta", "Prediccion(%)", "Imagen"])

# start the file video stream thread and allow the buffer to
# start to fill
if __name__ == "__main__":

	while True:
		try:
			print("[INFO] Estableciendo conexion con la camara {}".format(INPUT_FILE))
			#logging.info("[INFO] Estableciendo conexion con la camara {}".format(INPUT_FILE))
			fvs = FileVideoStream(INPUT_FILE).start()
			print("llego linea 68")
			if fvs.stream.isOpened(): 
				print("Flujo abierto continuar")
				break #  Se rompe la espera de conexion de la camara
			time.sleep(5)
		except RuntimeError as e:
			print("llego linea 71")
			print("[INFO] Error  al leer camara remota, se vuelve a intentar...", sys.exc_info()[0])
			print(e)
			time.sleep(5)
		except:
			print("llego linea 76")
			print("[INFO] Error general al leer camara remota, se vuelve a intentar...")
			time.sleep(5)
	#Configuracion del modelo
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
	options = {"CAP_PROP_FPS":15}


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
	frameRate = 15 # Frame number interception interval (one frame is intercepted every 100 frames)
    
    #Extra dimensiones para las capturas
	sceneExtra = 50

	time.sleep(1.0)

	# start the FPS timer
	print("llego linea 128")
	fps = FPS().start()
	print("llego linea 130")

	# loop over frames from the video file stream
	while fvs.more():
		print("llego linea 134")
		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale (while still retaining 3
		# channels)
		frame = fvs.read()
		print("llego linea 139")
		if frame is None:
			break
		
		print("llego linea 140")
		frame = imutils.resize(frame, width=450)
		print("llego linea 141")
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		print("llego linea 142")
		frame = np.dstack([frame, frame, frame])
		print("llego linea 143")

		# display the size of the queue on the frame
		cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
			(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
		
		print("llego linea 144")

		fps.update()
		# show the frame and update the FPS counter
		 # Show output window, siempre que este activado
		print("SHOW_IMAGE {}".format(SHOW_IMAGE))
		print("llego linea 166")
		if SHOW_IMAGE:
			print("llego linea 167")
			cv2.imshow("Frame", frame)
		# check for 'q' key if pressed
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
		

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	fvs.stop()
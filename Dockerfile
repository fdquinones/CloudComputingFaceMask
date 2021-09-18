FROM python:3.9-bullseye AS base
EXPOSE 5000

WORKDIR /utpl
COPY . .


RUN python --version
RUN ls
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip install opencv-python==4.1.2.30
RUN pip install vidgear
RUN pip install imutils
RUN pip install firebase_admin
RUN pip install psutil
RUN pip install paho-mqtt


CMD [ "python", "appV10.py"]

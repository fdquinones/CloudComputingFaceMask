FROM tensorflow/tensorflow AS base
EXPOSE 80

WORKDIR /utpl
COPY . .


RUN python --version
RUN ls
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip install chrysalis
RUN pip install imutils
RUN python mask_detection.py

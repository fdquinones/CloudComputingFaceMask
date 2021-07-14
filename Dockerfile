FROM tensorflow/tensorflow AS base
EXPOSE 80

WORKDIR /utpl
COPY . .


RUN python --version
RUN ls
RUN pip install imutils
RUN python mask_detection.py

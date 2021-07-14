FROM tensorflow/tensorflow AS base
EXPOSE 80

WORKDIR /utpl
COPY . .


RUN python --version
RUN ls
RUN apt install libgl1-mesa-glx -y
RUN pip install opencv-python
RUN pip install chrysalis
RUN pip install imutils
RUN python mask_detection.py

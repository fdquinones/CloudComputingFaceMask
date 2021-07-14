FROM tensorflow/tensorflow AS base
EXPOSE 5000

WORKDIR /utpl
COPY . .


RUN python --version
RUN ls
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip install opencv-python
RUN pip install chrysalis
RUN pip install imutils
RUN pip install Flask

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]

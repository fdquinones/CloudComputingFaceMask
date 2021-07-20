from flask import Flask, render_template, Response
import cv2
import os

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    items = os.listdir('static/detections')
    print(items)
    items = ['detections/' + file for file in items]
    items.sort()
    return render_template('index.html', items = items)


if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, Response
import cv2
import chrysalis
import os

app = Flask(__name__)

port = '6113'
host = 'casabelen.monstrous-buildsergeant-3oljx.chrysvideo.com'
password = 'bbcb9c9cd13a765ad6b83f996884eb3c'
cert_path = 'chryscloud.cer'

chrys = chrysalis.Connect(host=host, port=port, password=password, ssl_ca_cert=cert_path)

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        img = chrys.VideoLatestImage()  # read the camera frame
        if img is None:
            break
        else:
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
    items = os.listdir('static/detections')
    hists = ['detections/' + file for file in items]
    return render_template('index.html', items = items)


if __name__ == '__main__':
    app.run(debug=True)
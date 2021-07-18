const Stream = require('node-rtsp-stream-jsmpeg')

const options = {
  name: 'mystream',
  url: 'rtsp://159.69.217.242:9665/mystream',
  wsPort: 3333
}

stream = new Stream(options)
stream.start()
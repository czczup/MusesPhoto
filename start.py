from flask import Response, Flask,send_from_directory
from flask import render_template, jsonify, request
from filter import Filter
from datetime import datetime
from utils import get_host_ip
import tensorflow as tf
import numpy as np
import base64
import json
import cv2
import model
import time
import reader


class TransferServer():
    def __init__(self):
        self.filters = {'oil_painting': Filter(name='oil_painting'),
                        'golden_time': Filter(name='golden_time'),
                        'ocean_heart': Filter(name='ocean_heart'),
                        'nature': Filter(name='nature'),
                        'old_photo': Filter(name='old_photo'),
                        'engraving_art': Filter(name='engraving_art'),
                        'feathers': Filter(name='feathers'),
                        'wave': Filter(name='wave'),
                        'starry_night': Filter(name='starry_night'),
                        'cubist': Filter(name='cubist'),
                        'mosaic': Filter(name='mosaic')}
        self.upload_path = "./upload/"

    def get_image_and_filter(self):
        if request.method=='POST':
            json_data = request.get_data().decode("utf-8")
            dict_data = json.loads(json_data)
            image_base64 = dict_data['image_base64']
            image = base64.b64decode(image_base64)
            image = np.fromstring(image, np.uint8)
            image_upload = cv2.imdecode(image, cv2.IMREAD_COLOR)
            filter_name = dict_data['filter_name']
            image_name = filter_name+"_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".jpg"
            cv2.imwrite(self.upload_path+image_name, image_upload)
            return image_name,filter_name

    def transfer(self):
        image_name, filter_name = self.get_image_and_filter()
        tf.logging.info("Load data completed.")
        image_save_path = self.filters[filter_name].style_transfer(image_name)
        tf.logging.info("Image style transfer completed.")
        return image_save_path



app = Flask(__name__)
transfer_server = TransferServer()

@app.route("/api",methods=['POST'])
def index():
    time1 = time.time()
    image_path = transfer_server.transfer()
    time2 = time.time()
    tf.logging.info("Using time: %.3fs"%(time2-time1))
    image_json = {
        'download_path':"http://120.79.162.134:80/"+image_path[2:]
    }
    print(image_json['download_path'])
    return json.dumps(image_json)
    # return image_json['download_path']

@app.route('/save/<filename>', methods=['GET'])
def download(filename):
    return send_from_directory('save',filename, as_attachment=True)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(host='0.0.0.0', port=80, threaded=True)


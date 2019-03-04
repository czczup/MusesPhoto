from flask import Flask, send_from_directory
from flask import request
from filter import Filter
import tensorflow as tf
from datetime import datetime
import numpy as np
from PIL import Image
from scipy import misc
import base64
import json
import cv2
import time
import os
import io


class TransferServer:
    def __init__(self):
        pb_files = os.listdir("models/")  # 获取所有滤镜的文件名
        self.filters = {}
        for filename in pb_files:
            self.filters[int(filename.split(".pb")[0])] = Filter(name=filename)  # 载入所有滤镜
            print("滤镜%s载入成功" % filename)

    def get_image_and_filter(self):
        time_ = time.time()
        file = request.files['file']
        upload_id = int(request.form['upload_id'])
        image = misc.imread(file)
        print("Upload file using time:", time.time() - time_)

        return image, upload_id

    def transfer(self):
        time_ = time.time()
        image, upload_id = self.get_image_and_filter()
        tf.logging.info("Load data completed, using time:%.3f" % (time.time()-time_))
        time_ = time.time()
        image = self.filters[upload_id].style_transfer(image)
        tf.logging.info("Transfer image completed, using time:%.3f" % (time.time()-time_))
        time_ = time.time()
        path = "images/" + str(upload_id) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        cv2.imwrite(path, image)
        tf.logging.info("IO, using time:%.3f" % (time.time()-time_))
        tf.logging.info("Image style transfer completed.")
        return path


app = Flask(__name__)


@app.route("/api/transfer", methods=['POST'])
def index():
    image_path = transfer_server.transfer()
    image_json = {
        'image': "http://art.deepicecream.com:7004" + image_path
    }
    return json.dumps(image_json)


@app.route('/images/<filename>', methods=['GET'])
def download(filename):
    return send_from_directory('images', filename, as_attachment=True)


if __name__ == '__main__':
    deviceId = input("please input device id (0-3): ")
    os.environ["CUDA_VISIBLE_DEVICES"] = deviceId
    tf.logging.set_verbosity(tf.logging.INFO)
    transfer_server = TransferServer()
    app.run(host='0.0.0.0', port=7004, threaded=True)

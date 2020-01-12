from flask import Flask, send_from_directory
from flask import request, Response
from flask_cors import CORS
from filter import Filter
import tensorflow as tf
from datetime import datetime
from scipy import misc
import requests
import json
import cv2
import time
import os
import skimage


class TransferServer:
    def __init__(self):
        pb_files = os.listdir("models/")  # 获取所有滤镜的文件名
        self.filters = {}
        for filename in pb_files:
            self.filters[int(filename.split(".pb")[0])] = Filter(name=filename)  # 载入所有滤镜
            print("滤镜%s载入成功" % filename)

    def get_image_and_filter(self):
        file = request.files['file']
        upload_id = int(request.form['upload_id'])
        image = misc.imread(file)[..., 0:3] # 若图像通道数多于3，则取前三通道
        print(image.shape)
        # image = skimage.util.random_noise(image, mode='gaussian', seed=None, clip=True, var=(20/255.0)**2)
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
        cv2.imwrite(path, image[..., ::-1]) # BGR转RGB
        tf.logging.info("IO, using time:%.3f" % (time.time()-time_))
        tf.logging.info("Image style transfer completed.")
        return path


app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route("/api/transfer", methods=['POST'])
def index():
    image_path = transfer_server.transfer()
    image_json = {
        'image': "http://art.deepicecream.com:7004/" + image_path
    }
    # requests.get("http://muses.deepicecream.com:7010/api/filter/use/"+request.form['upload_id'])
    print(image_json['image'])
    return json.dumps(image_json)


@app.route("/api/addFilter/<filter_id>", methods=['POST'])
def add_filter(filter_id):
    filter_id = int(filter_id)
    transfer_server.filters[filter_id] = Filter(name=str(filter_id)+".pb")  # 载入所有滤镜
    print("添加新滤镜成功，滤镜id为%d" % filter_id)
    return Response(response='success', status=200, content_type='text/html;charset=utf8')


@app.route('/images/<filename>', methods=['GET'])
def download(filename):
    return send_from_directory('images', filename, as_attachment=True)


if __name__ == '__main__':
    deviceId = input("please input device id (0-3): ")
    os.environ["CUDA_VISIBLE_DEVICES"] = deviceId
    tf.logging.set_verbosity(tf.logging.INFO)
    transfer_server = TransferServer()
    app.run(host='0.0.0.0', port=7004, threaded=True)

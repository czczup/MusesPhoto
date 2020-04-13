from flask import Flask, send_from_directory
from flask import request, Response
from flask_cors import CORS
from flask import request
from filter import Filter
import tensorflow as tf
from datetime import datetime
from scipy import misc
from PIL import Image
import numpy as np
import json
import cv2
import time
import os


class TransferServer:
    def __init__(self):
        pb_files = os.listdir("models/")  # 获取所有滤镜的文件名
        self.filters = {}
        for filename in pb_files:
            self.filters[int(filename.split(".pb")[0])] = Filter(name=filename)  # 载入所有滤镜

    def get_image_and_filter(self):
        file = request.files['file']
        upload_id = int(request.form['upload_id'])
        image = misc.imread(file)[..., 0:3] # 若图像通道数多于3，则取前三通道
        image = Image.fromarray(image)
        width, height = image.size
        multiplier = ((width*height)/(1024**2))**0.5 # 图书的缩放系数
        if multiplier > 1: # 对图像进行缩放
            image = image.resize((int(width/multiplier), int(height/multiplier)))
        image = np.array(image)
        tf.logging.info("Image size: (%d, %d)" % (image.shape[0],image.shape[1]))
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
    """ 风格迁移API """
    image_path = transfer_server.transfer()
    image_json = {
        'image': "http://gpu.vanxnf.top:10004/" + image_path
    }
    # 统计滤镜使用次数
    # requests.get("http://muses.deepicecream.com:7010/api/filter/use/"+request.form['upload_id'])
    print(image_json['image'])
    return json.dumps(image_json)


@app.route("/api/addFilter/<filter_id>", methods=['POST'])
def add_filter(filter_id):
    """ 添加滤镜API """
    filter_id = int(filter_id)
    transfer_server.filters[filter_id] = Filter(name=str(filter_id)+".pb")  # 载入所有滤镜
    print("添加新滤镜成功，滤镜id为%d" % filter_id)
    return Response(response='success', status=200, content_type='text/html;charset=utf8')


@app.route('/images/<filename>', methods=['GET'])
def download(filename):
    """ 下载图像API """
    return send_from_directory('images', filename, as_attachment=True)


@app.route('/download/<file>', methods=['GET'])
def download_page(file):
    html = "<html><img src=http://art.deepicecream.com:7004/images/"+file+".jpg style='width:100%' ></html>"
    return html

@app.route('/api/upload', methods=['POST'])
def upload_file():
    f = request.files['file']
    upload_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    f.save("images/"+upload_time+".jpg")
    image_json = {
        'image': "http://art.deepicecream.com:7004/download/"+upload_time
    }
    print(image_json['image'])
    return json.dumps(image_json)



if __name__ == '__main__':
    deviceId = input("please input device id (0-3): ")
    os.environ["CUDA_VISIBLE_DEVICES"] = deviceId
    tf.logging.set_verbosity(tf.logging.INFO)
    transfer_server = TransferServer()
    app.run(host='0.0.0.0', port=7004, threaded=True)
    # 端口为7004

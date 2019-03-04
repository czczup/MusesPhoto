import requests
from PIL import Image
import base64
import json
import urllib
import socket
from utils import get_host_ip

filters = ['oil_painting','golden_time',
           'ocean_heart','nature',
           'night_scene','engraving_art',
           'feathers','wave',
           'starry_night','cubist']
f = open('./img/flower1.jpg','rb') #二进制方式打开图文件
image_base64 = base64.b64encode(f.read()) #读取文件内容，转换为base64编码
image_base64 = image_base64.decode('UTF-8')
f.close()

# post发送的数据
for filter in filters:
    postData = {
        'image_base64':image_base64,
        'filter_name':filter,
        'image_width':256,
        'image_height':256
    }

    postData = json.dumps(postData).encode()
    response = requests.post('http://'+get_host_ip()+':80/api',data=postData)
    # response = requests.post('http://'+'120.79.162.134'+':80/',data=postData)
    print(response)

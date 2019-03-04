import requests
import base64
import json
import time
import os

time1 = time.time()
f = open('test/test5.jpg', 'rb')  # 二进制方式打开图文件
image_base64 = base64.b64encode(f.read())  # 读取文件内容，转换为base64编码
image_base64 = image_base64.decode('UTF-8')
f.close()
print("using time:", time.time() - time1)

# post发送的数据
ids = [int(file.split('.pb')[0]) for file in os.listdir("models/")]

for id_ in ids:
    postData = {
        'image_base64': image_base64,
        'upload_id': id_
    }

    postData = json.dumps(postData).encode()
    response = requests.post('http://art.deepicecream.com:7004/api/transfer', data=postData)
    print(response)


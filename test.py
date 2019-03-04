import requests
import base64
import json
import time
import os

time1 = time.time()
files = {'file': ('test5.jpg', open('test/test5.jpg', 'rb'), 'image/jpg')}
print("using time:", time.time() - time1)

# post发送的数据
ids = [int(file.split('.pb')[0]) for file in os.listdir("models/")]

for id_ in ids:
    postData = {
        'upload_id': id_
    }
    # postData = json.dumps(postData)
    response = requests.post('http://art.deepicecream.com:7004/api/transfer', data=postData, files=files)
    print(response)
    time.sleep(3)

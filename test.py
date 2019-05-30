import requests
import base64
import json
import time
import os

time1 = time.time()
files = {'file': ('test2.jpg', open('test/test5.jpg', 'rb'), 'image/jpg')}
print("using time:", time.time() - time1)

postData = {
    'upload_id': 497
}
# postData = json.dumps(postData)
response = requests.post('http://art.deepicecream.com:7004/api/transfer', data=postData, files=files)
print(response)

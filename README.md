# MusesPhoto部署指南

## 1 部署环境
```sh
conda install python=3.6
conda install cudatoolkit=9.0
conda install cudnn==7.1.2
conda install tensorflow-gpu==1.9
conda install pillow
conda install opencv
conda install flask
conda install flask_cors
conda install scipy==1.1.0
conda install requests
conda install numpy==1.16.0
```
## 2 服务器启动
```sh
python start.py
```
## 3 API文档
### 3.1 添加滤镜
- 地址: `/api/addFilter/<filter_id>`
- 方法: `POST`
### 3.2 下载图片
- 地址: `/images/<filename>`
- 方法: `GET`
### 3.3 风格迁移
- 地址: `/api/transfer`
- 方法: `POST`
- 参数: 
  - 文件: 使用`multipart/form-data`上传图像，名称为`file`
  - `upload_id`: 滤镜信息中的`upload_id`
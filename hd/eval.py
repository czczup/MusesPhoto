import tensorflow as tf
from PIL import Image
import os
import numpy as np
import time
import filter as f

os.environ["CUDA_VISIBLE_DEVICES"] = str(input('选择GPU:'))
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.logging.set_verbosity(tf.logging.INFO)


class ImageItem:
    def __init__(self, id, image):
        self.id = id
        self.image = image


def fill_image(image):
    width, height = image.size
    print('图片宽高为%d*%d'%(width, height))
    new_length = width if width>height else height
    new_image = Image.new(image.mode, (new_length, new_length), color='white')
    if width > height:
        new_image.paste(image, (0, int((new_length-height)/2)))
    else:
        new_image.paste(image, (int((new_length-width)/2), 0))

    return new_image


def padding(image, padding):
    width, height = image.size
    new_image = Image.new(image.mode, (width+2*padding, height+2*padding), color='white')
    new_image.paste(image, (padding, padding, width+padding, height+padding))
    return new_image


def crop_image(image, part, padding):
    width, height = image.size
    item_length = int((width-2*padding)/part)
    box_list = []
    count = 0
    for r in range(0, part):
        for c in range(0, part):
            count += 1
            x_left = c*item_length
            y_left = r*item_length
            x_right = (c+1)*item_length+2*padding
            y_right = (r+1)*item_length+2*padding
            box = [x_left, y_left, x_right, y_right]
            box_list.append(box)
    print('共切割成%d块'%count)
    image_list = [image.crop(box) for box in box_list]
    return image_list


def merge_image(image_list, part):
    image_list = [image_list[i:i+part] for i in range(0, len(image_list), part)]

    width, height = image_list[0][0].size
    print("图片的像素为%d*%d" % (width*part, height*part))
    # 将多切的组合成一个更大的图片
    target = Image.new('RGB', (width*part, height*part))
    # 每一行
    for i, row in enumerate(image_list):
        for j, item in enumerate(row):
            a = j*width
            b = i*height
            c = a+width
            d = b+height
            target.paste(item, (a, b, c, d))
    return target


def unpadding(image, padding):
    width, height = image.size
    box = (padding, padding, width-padding, height-padding)
    image = image.crop(box)
    return image


def optimize_seam(image_list, part, seam, padding):
    image_list = [image_list[i:i+part] for i in range(0, len(image_list), part)]
    width, height = [item-2*seam for item in image_list[0][0].size]
    width_padding, height_padding = image_list[0][0].size
    print(width)
    print("图像的像素为: ", width*part, "*", width*part)
    target = Image.new('RGB', (width*part+2*seam, width*part+2*seam))
    for i, row in enumerate(image_list):
        for j, item in enumerate(row):
            a = j*width+seam  # 图片距离左边的大小
            b = i*width+seam  # 图片距离上边的大小
            c = a+width-2*seam  # 图片距离左边的大小 + 图片自身宽度
            d = b+width-2*seam  # 图片距离上边的大小 + 图片自身高度
            # print((a,b,c,d, width_padding))
            image_unpadding = unpadding(item, padding=2*seam)
            # print(image_unpadding.size)
            target.paste(image_unpadding, (a, b, c, d))
            if j!=0:
                start_x = a-2*seam
                end_x = a
                start_y = b-seam
                end_y = d+seam
                for x in range(start_x, end_x):
                    for y in range(start_y, end_y):
                        alpha = (2*seam-(x-start_x))/(2*seam)
                        pixel1 = image_list[i][j-1].getpixel(((width_padding-2*seam-1)+(x-start_x), y-start_y+seam))
                        pixel2 = image_list[i][j].getpixel((x-start_x, y-start_y+seam))
                        pixel_add = tuple([int(pixel1[i]*alpha+pixel2[i]*(1-alpha)) for i in range(3)])
                        target.putpixel((x, y), pixel_add)
                    # else:
                    #     print("x:%d" % x)
            if i != 0:
                start_x = a-seam
                end_x = c+seam
                start_y = b-2*seam
                end_y = b
                for y in range(start_y, end_y):
                    for x in range(start_x, end_x):
                        alpha = (2*seam-(y-start_y))/(2*seam)
                        pixel1 = image_list[i-1][j].getpixel(
                            (x-start_x+seam, (width_padding-2*seam-1)+(y-start_y)))
                        pixel2 = image_list[i][j].getpixel((x-start_x+seam, y-start_y))
                        pixel_add = tuple([int(pixel1[i]*alpha+pixel2[i]*(1-alpha)) for i in range(3)])
                        target.putpixel((x, y), pixel_add)
                    else:
                        print("x:%d" % x)
    return target


if __name__=='__main__':
    style_model = '139.pb'
    filter = f.Filter(name=style_model)
    part = 100
    big_cut = 10
    pad = 40
    seam = 14
    image_list = [
        'England.jpg'
    ]

    for idx, filename in enumerate(image_list):
        image = Image.open(filename)
        image_name = filename.split('.')[0].split('/')[-1]
        model_name = style_model.split('.')[0].split('/')[-1]
        image = fill_image(image)
        image = padding(image, padding=pad)
        print('1.1：切块')
        image_list = crop_image(image, part=part, padding=pad)
        print('>>> 一共切割成%d块'%len(image_list))
        item_list = [ImageItem(id, image) for id, image in enumerate(image_list)]
        print('1.2：乱序')
        np.random.shuffle(item_list)
        print('1.3：合并图像')
        image = merge_image([item.image for item in item_list], part=part)
        image = padding(image, padding=pad)
        print('2.1：分区')
        image_list = crop_image(image, part=big_cut, padding=pad)
        print('2.2：风格迁移')
        temp_list = []
        temp_list_origin = []
        for j, image in enumerate(image_list):
            print("正在风格化第%d/%d块"%(j+1, big_cut*big_cut))
            start_time = time.time()
            result = filter.style_transfer(image)
            end_time = time.time()
            tf.logging.info('Elapsed time: %fs'%(end_time-start_time))
            im = Image.fromarray(np.uint8(result))
            temp_list.append(im)

        print('2.3：合并图像')
        temp_list = [unpadding(image, padding=pad) for image in temp_list]
        image = merge_image(temp_list, part=big_cut)
        image.save("random.jpg")
        print("3.1：切块")
        image_list = crop_image(image, part=part, padding=0)
        for index, image in enumerate(image_list):
            item_list[index].image = unpadding(image, padding=pad-seam)  # 用风格化的图像替换原图像
        print("3.2：恢复原始顺序")
        item_list = sorted(item_list, key=lambda item: item.id)
        image_list = [item.image for item in item_list]
        print("3.3：合并图像")
        image = optimize_seam(image_list, part=part, seam=seam, padding=pad)
        image = unpadding(image, padding=20)
        print('开始保存图片')
        image.save("shuffle.jpg")

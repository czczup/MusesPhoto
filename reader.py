import tensorflow as tf

def get_image(path):
    img_bytes = tf.read_file(path)
    image = tf.image.decode_jpeg(img_bytes, channels=3)
    return image

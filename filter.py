import tensorflow as tf
import model

class Filter():
    def __init__(self, name=None):
        self.name = name
        self.model_path = "./model/"+name+".ckpt-done"
        self.upload_path = "./upload/"
        self.save_path = "./save/"
        self.graph = None
        self.sess = None
        self.input = None
        self.output = None
        self.load_model()
        tf.logging.set_verbosity(tf.logging.INFO)
        
    def load_model(self):
        self.graph = tf.Graph()
        tf.logging.info('Create the graph for %s successfully.'%self.name)
        self.sess = tf.InteractiveSession(graph=self.graph)
        tf.logging.info('Create the session for %s successfully.'%self.name)
        with self.sess.graph.as_default():
            with self.sess.as_default():
                image_placeholder = tf.placeholder(tf.float32, [None, None, 3], name='image')
                # Add batch dimension
                image = tf.expand_dims(image_placeholder, 0)
                generated = model.net(image, training=False)
                generated = tf.cast(generated, tf.uint8)
                # Remove batch dimension
                generated = tf.squeeze(generated, [0])
                saver = tf.train.Saver(tf.global_variables())
                self.input = image_placeholder
                self.output = generated
                tf.logging.info("Record the input and output of %s successfully."%self.name)
                saver.restore(self.sess, self.model_path)
                tf.logging.info("Restore the model of %s successfully."%self.name)
    
    def style_transfer(self,image_name):
        sess = self.sess
        with sess.as_default():
            with sess.graph.as_default():
                with open(self.save_path+image_name, 'wb') as image:
                    image_upload = sess.run(self.get_image(self.upload_path+image_name))
                    result = sess.run(tf.image.encode_jpeg(self.output), feed_dict={self.input: image_upload})
                    image.write(result)
                return self.save_path+image_name

    def get_image(self, path):
        img_bytes = tf.read_file(path)
        image = tf.image.decode_jpeg(img_bytes, channels=3)
        return image

    def __del__(self):
        self.sess.close()
    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    filter = Filter(name='oil_painting')
import tensorflow as tf
import os
import time

class Filter:
    def __init__(self, name=None):
        self.name = name
        self.model_path = "models/" + name
        self.graph = None
        self.sess = None
        self.input = None
        self.output = None
        self.load_model()
        tf.logging.set_verbosity(tf.logging.INFO)

    def load_model(self):
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.graph)
        tf.logging.info('Create the session for %s successfully.' % self.name)
        with self.sess.graph.as_default():
            with self.sess.as_default():
                tf.global_variables_initializer().run()
                with tf.gfile.FastGFile(self.model_path, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')
                    self.input = self.graph.get_tensor_by_name("input:0")
                    self.output = self.graph.get_tensor_by_name("Squeeze:0")
                    # for op in self.graph.get_operations():
                    #     print(op.name)

    def style_transfer(self, image):
        sess = self.sess
        with sess.as_default():
            with sess.graph.as_default():
                result = sess.run(self.output, feed_dict={self.input: [image]})
        return result

    def __del__(self):
        self.sess.close()


if __name__ == '__main__':
    deviceId = input("please input device id (0-3): ")
    os.environ["CUDA_VISIBLE_DEVICES"] = deviceId
    tf.logging.set_verbosity(tf.logging.INFO)
    filter = Filter(name='252.pb')

import tensorflow as tf
import cv2
import numpy as np
import argparse
import os

from openpose_pipeline.common import estimate_pose, draw_humans, preprocess


tf.debugging.set_log_device_placement(True)

#from the tensorflow docs: (https://www.tensorflow.org/guide/migrate)
# """There is no straightforward way to upgrade a raw Graph.pb file to
#    TensorFlow 2.0.
#    Your best bet is to upgrade the code that generated the file."""
#hey, thanks a lot.
#"""But, if you have a "Frozen graph" (a tf.Graph where the variables
#   have been turned
#   into constants), then it is possible to convert this to a concrete_function using
#   v1.wrap_function:"""
#ok, let's try that.
def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")
    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def tf_preprocess(images,width,height):
    images = images[...,::-1] #convert bgr to rgb
    images = tf.image.resize(images,[height,width],antialias=True)
    images = tf.cast(images,tf.float32)
    images = images * (2.0 / 255) - 1.0
    return images
    
  

class HumanPoseEstimator(object):
    def __init__(self,model_file=None,input_width=656,input_height=368,batch_size=1):
        from tensorflow.core.framework import graph_pb2
        graph_def = graph_pb2.GraphDef()
        if model_file==None:
            thisDir = os.path.dirname(os.path.abspath(__file__))
            model_file = os.path.sep.join((thisDir,
                                           "openpose_pipeline",
                                           "models",
                                           "optimized_openpose.pb"))           
        with open(model_file,'rb') as f:
            graph_def.ParseFromString(f.read())
        #self.inputs,self.heatmaps_tensor,self.pafs_tensor = tf.import_graph_def(
        #     graph_def,
        #     return_elements=['inputs:0',
        #                      'Mconv7_stage6_L2/BiasAdd:0',
        #                      'Mconv7_stage6_L1/BiasAdd:0'])
        #self._model = tf.keras.Model(inputs=self.inputs,outputs=
        #                                                     [self.heatmaps_tensor,
        #                                                         self.pafs_tensor])
        #the above can't be done :( and looks like the tf authors have abandoned us.
        
        graph_func = wrap_frozen_graph(graph_def,
                                       inputs="inputs:0",
                                       outputs=['Mconv7_stage6_L2/BiasAdd:0',
                                                'Mconv7_stage6_L1/BiasAdd:0'])
        inputs = tf.keras.Input(shape=(input_height,input_width,3),dtype=tf.float32,
                                batch_size=batch_size)

        #graph_func = tf.function(graph_func)
        
        #outs = graph_func(inputs)

        #self.model = tf.keras.Model(inputs=inputs,outputs=outs)

        self.graph_func = graph_func
        
        self.input_width=input_width
        self.input_height=input_height
        

    @tf.function
    def process_raw(self,imgs):
        """
        imgs: array or list of BGR images of the type read in by opencv

        returns: heatmaps, pafs
        """
        #TODO this can be made more efficient by replaceing preprocess() with math
        #that works with 4-dimensional tensors.
        #imgs = [preprocess(img,self.input_width,self.input_height) for img in imgs]
        #                                                            #see note above
        imgs = tf_preprocess(imgs,self.input_width,self.input_height)
        #imgs = np.concatenate(imgs,axis=0)#see note above
        #imgs = tf.constant(imgs,dtype="float32")
        heatMat,pafMat=self.graph_func(imgs)
        #heatMat,pafMat = self.model(imgs,training=False)
        return heatMat,pafMat

    def get_humans(self,imgs=None,maps=None):
        if imgs is not None and type(imgs) is not np.ndarray:
            imgs = np.array(imgs)
        if maps is None:
            maps = self.process_raw(imgs)
        maps = [m.numpy() for m in maps] 
        humanss = [estimate_pose(heatMat,pafMat) for heatMat,pafMat in zip(*maps)]
        #there is a signfificant amount of engineering in estimate_pose. it is
        #written in python and looks well-written. based on how it works, it
        #looks like it can't be parallelized or ported to tf easily.
        #with a lot of work, I could use heatmaps and pafs over several frames,
        #maybe, to do a better job of estimating pose in a single frame.
        #another possibility is to just use some simple smoothing/regression
        return humanss

    def draw_humans(self,imgs,humans_per_img=None):
        if humans_per_img is None:
            humans_per_img = self.get_humans(imgs)
        return [draw_humans(img,humans) for humans,img in zip(humans_per_img,imgs)]


def test1():
    h = HumanPoseEstimator()
    i = [cv2.imread("./openpose_pipeline/humans.jpeg")]*10
    import time
    t = time.time()
    for j in range(10):
        h.process_raw(i)
        print(time.time()-t)
        t = time.time()

def test2():
    h = HumanPoseEstimator()
    i = [cv2.imread("./openpose_pipeline/humans.jpeg")]*10
    d = h.draw_humans(i)
    for dd in d:
        cv2.imshow("derp",dd)
        cv2.waitKey()

def test3():
    h = HumanPoseEstimator()
    i = [cv2.imread("./openpose_pipeline/black-test-1.jpeg")]
    humanss = h.get_humans(i)
    print(humanss)
    d = h.draw_humans(i,humans_per_img=humanss)
    for dd in d:
        cv2.imshow("derp",dd)
        cv2.waitKey()

if __name__=='__main__':
    test3()
        
        

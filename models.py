from glob import glob
import os
import tensorflow as tf

import logging

logging.basicConfig(level=logging.DEBUG)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'models')


def load_v2_savedmodel(path):
    return tf.saved_model.load(path).signatures['default']

def load_from_saved_model_subdir(path):
    return tf.saved_model.load(os.path.join(path,'saved_model'))

def load_v1_sseg(path):
    g = load_v1_frozengraph(path)
    return wrap_frozen_graph(g,
                             inputs='ImageTensor:0',
                             outputs='SemanticPredictions:0')

MODEL_NAMES = { #name : (loader, url)
    'deeplabv3_pascal_trainval':(load_v1_sseg, #semantic seg
            'http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz'),
    'xception71_dpc_cityscapes_trainval':(load_v1_sseg, #semantic seg
                'http://download.tensorflow.org/models/deeplab_cityscapes_xception71_trainvalfine_2018_09_08.tar.gz'),
    'deeplabv3_xception_ade20k_train':(load_v1_sseg,'http://download.tensorflow.org/models/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz'),
    'coco_centernet':(load_from_saved_model_subdir,# detections and keypoints
            'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_1024x1024_kpts_coco17_tpu-32.tar.gz')
    }


def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")
    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

def load_v1_frozengraph(path):
    pbs = glob(os.path.join(path,'*.pb'))
    if len(pbs)!=1:
        raise FileNotFoundError(os.path.join(path,'*.pb')+' [exactly one]')
    frz = list(pbs)[0]
    from tensorflow.core.framework import graph_pb2
    graph_def = graph_pb2.GraphDef()
    with open(frz,'rb') as f:
        graph_def.ParseFromString(f.read())
    return graph_def


def fetch_model(local_name):
    url = MODEL_NAMES[local_name][1]
    model_tgz = url.split('/')[-1]
    model_path = os.path.join(MODELS_DIR,local_name)
    if os.path.exists(model_path):
        return model_path
    else:
        logging.info("fetching %s"%url)
        os.system("cd {mdir} && wget {url}".format(
            mdir=MODELS_DIR,url=url))
        old_dirs = glob(os.path.join(MODELS_DIR,'*'))
        logging.info("extracting %s"%model_tgz)
        os.system("cd {mdir} && tar xzvf {arch}".format(
            mdir=MODELS_DIR,arch=model_tgz))
        new_dirs = glob(os.path.join(MODELS_DIR,'*'))
        new_dir = set(new_dirs) - set(old_dirs)
        assert len(new_dir)==1
        new_dir = list(new_dir)[0]
        if os.path.split(new_dir)[-1] != local_name:
            logging.info("renaming {created} to {lname}".format(
                created=new_dir,lname=local_name))
            os.system("cd {mdir} && rm -rf {lname} && mv {created} {lname}".format(
                mdir=MODELS_DIR,
                created=os.path.split(new_dir)[-1],
                lname=local_name))
        return model_path

def load(local_name):
    path = fetch_model(local_name)
    loader = MODEL_NAMES[local_name][0]
    return loader(path)

def test():
    import numpy as np
    model = load('deeplabv3_pascal_trainval')
    print(type(model))
    print(dir(model))
    return model
    #out = model(tf.zeros((1,360,640,3),tf.float32))
    #print (list(out.keys()))
    #for key in out.keys():
    #    print(key,out[key].shape)

if __name__=='__main__':
    test()

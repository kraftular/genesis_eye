import pyzed.sl as sl
import cv2
import numpy as np
from tqdm import tqdm
import tables
import inspect
import os

import logging
logging.basicConfig(level=logging.DEBUG)

class ZedException(Exception):
    pass

FDATA_GROUP = 'processed_video_data'#earray data
OBJ_GROUP   = 'objects'#for tables of global vars, etc.

DEFAULT_COMPRESSION_LEVEL=0

VIDEOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'videos')

DTYPE_ATOM_MAP = {
    np.float32().dtype     :tables.Float32Atom,
    np.float64().dtype     :tables.Float64Atom,
    np.uint8().dtype       :tables.UInt8Atom,
    np.uint16().dtype      :tables.UInt16Atom,
    np.int32().dtype       :tables.Int32Atom,
    np.int64().dtype       :tables.Int64Atom,
}


def get_atom(np_example,use_shape=True):
    if use_shape:
        print(np_example.shape)
        return DTYPE_ATOM_MAP[np_example.dtype](np_example.shape)
    else:
        return DTYPE_ATOM_MAP[np_example.dtype]((1,))
    

def recreate_earray(where,name,np_example,h5file,complib,complevel):
    exist_nodes = h5file.list_nodes(where)
    if any(node.name==name for node in exist_nodes): 
        logging.warning("destroying and recreating %s"%name)
        h5file.remove_node(where,name)
        h5file.flush()
    atom = get_atom(np_example)
    #print(atom)
    filters = tables.Filters(complevel=complevel, complib=complib)
    earray = h5file.create_earray(where,name,atom,
                                  (0,),expectedrows=5000,
                                  filters=filters)
    return earray

def initialize_earrays(d,h5file,complib,complevel):
    return {name:recreate_earray('/'+FDATA_GROUP,name,d[name],h5file,
                                 complib=complib,complevel=complevel)
            for name in d}

def ensure_init(h5file):
    exist_nodes = h5file.list_nodes('/')
    if not any(node._v_name==FDATA_GROUP for node in exist_nodes):
        h5file.create_group('/',FDATA_GROUP)
    if not any(node._v_name==OBJ_GROUP for node in exist_nodes):
        h5file.create_group('/',OBJ_GROUP)

def process_from_generator(gen,h5file,complib,complevel,length=None):
    ensure_init(h5file)
    #extract one entry to use as template:
    d = next(gen)
    earrays = initialize_earrays(d,h5file,complib=complib,complevel=complevel)
    for name in d:
        earrays[name].append(d[name][np.newaxis,...])
        earrays[name].flush()
    if length is None:
        for d in tqdm(gen):
            for name in d:
                earrays[name].append(d[name][np.newaxis,...])
    else:
        for i in tqdm(range(length-1)):
            d = next(gen)
            for name in d:
                earrays[name].append(d[name][np.newaxis,...])
    for name in earrays:
        earrays[name].flush()
    h5file.flush()
        
def process_from_svo(svo_fname,extractor,h5file,
                     complib='zlib',complevel=DEFAULT_COMPRESSION_LEVEL):
    zed = sl.Camera()
    input_type = sl.InputType()
    input_type.set_from_svo_file(svo_fname)
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    init_params.sdk_verbose = False
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise ZedException("%r"%err)
    logging.info("open zed success")
    gen = extractor(zed)
    process_from_generator(gen,h5file,complib=complib,complevel=complevel)

def process_from_h5file(h5file,extractor,
                        complib='zlib',complevel=DEFAULT_COMPRESSION_LEVEL):
    gen = extractor(h5file)
    process_from_generator(gen,h5file,complib=complib,complevel=complevel,
                           length=get_max_array_len(h5file))

def get_array(h5file,name):
    where = '/%s/'%FDATA_GROUP
    earray = h5file.get_node(where,name,classname='EArray')
    return earray

def get_max_array_len(h5file):
    where = '/%s/'%FDATA_GROUP
    nodes = h5file.list_nodes(where)
    mx=0
    for node in nodes:
        mx = max(mx,len(node))
    return mx
    
    
def h5_extractor(row_processor,frame_args=None,array_args=None):
    """
    row_processor receives whole array for names in array_args. otherwise
    receives one frame's array. frame args is guessed from argspec if None.
    """
    if array_args is None:
        array_args = []
    if frame_args is None:
        frame_args = [name for name in inspect.getfullargspec(row_processor)[0]
                      if name not in array_args]
    def extractor(h5file):
        arrays = {name:get_array(h5file,name) for name in frame_args+array_args}
        for i in range(get_max_array_len(h5file)):
            fargs = {name:arrays[name][i] for name in frame_args}
            aargs = {name:arrays[name] for name in array_args}
            yield row_processor(**fargs,**aargs)
    return extractor


def get_hdf5_path(svo_name):
    if '/' not in svo_name:
        svo_name = os.path.join(VIDEOS_DIR,svo_name)
    if svo_name.lower().endswith('.svo'):
        svo_name = svo_name[:-4]
    return svo_name+'.hdf5'




#####################################################################################
#debugging and tests etc below this line

def test_svo_extractor_l(zed):
    mat = sl.Mat()
    runtime = sl.RuntimeParameters()
    while True:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(mat,sl.VIEW.LEFT)
            img = mat.get_data()[...,:3]
            yield dict(left_image=img.astype(np.uint8))
        else:
            break

def test_svo_extractor_lr(zed):
    mat = sl.Mat()
    runtime = sl.RuntimeParameters()
    while True:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(mat,sl.VIEW.LEFT)
            imgl = mat.get_data()[...,:3].copy()
            zed.retrieve_image(mat,sl.VIEW.RIGHT)
            imgr = mat.get_data()[...,:3].copy()
            yield dict(left_image=imgl.astype(np.uint8),
                       right_image=imgr.astype(np.uint8))
        else:
            break

def test_svo_extractor_blank(zed):
    mat = sl.Mat()
    runtime = sl.RuntimeParameters()
    while True:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(mat,sl.VIEW.LEFT)
            img = np.zeros_like(mat.get_data()[...,:3])
            yield dict(left_image=img.astype(np.uint8))
        else:
            break

@h5_extractor
def test_row_processor(left_image):
    return {'left_image_plus_5':left_image+5}


def test0():
    svo_file = './videos/HD720_SN27165053_16-29-04.svo'
    h5file = tables.open_file('test2.h5','w')
    process_from_svo(svo_file,test_svo_extractor_l,h5file)

def test1():
    svo_file = './videos/HD720_SN27165053_16-29-04.svo'
    h5file = tables.open_file('test2.h5','w')
    process_from_svo(svo_file,test_svo_extractor_lr,h5file)
    h5file.close()
    h5file = tables.open_file('test2.h5','a')
    process_from_svo(svo_file,test_svo_extractor_l,h5file)#should only recreate l

def test2():
    svo_file = './videos/HD720_SN27165053_16-29-04.svo'
    h5file = tables.open_file('test2.h5','w')
    process_from_svo(svo_file,test_svo_extractor_blank,h5file)

def test3():
    svo_file = './videos/HD720_SN27165053_16-29-04.svo'
    h5file = tables.open_file('test2.h5','a')
    #process_from_svo(svo_file,test_svo_extractor_lr,h5file)
    extractor = test_row_processor#h5_extractor(test_row_processor)
    process_from_h5file(h5file,extractor)

        
if __name__=='__main__':
    test3()
    

    
    

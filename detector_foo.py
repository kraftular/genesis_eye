###################################################################################
# detector foo: utilities for working with pretrained tensorflow object detectors #
#                                                                                 #
# adk, 9/20                                                                       #
###################################################################################



import tensorflow as tf
import numpy as np
import cv2
import tensorflow_hub as hub
import os
import threading
import atexit
import time
import sys

from coco import coco,pascal

from ade20k import ade20k_to_coco

import logging
logging.basicConfig(level=logging.DEBUG)

from models import MODELS_DIR

ADK_SSD = "ssd_v2_mobilenet_adk" #a failed attempt at making a framerate obj detector

#tf.debugging.set_log_device_placement(True)

DETECTOR_NAMES = {
    'openimages_detailed':(1,'faster_rcnn_openimages_v4_inception_resnet_v2/'),
    'coco_centernet':(2,'centernet_hg104_1024x1024_kpts_coco17_tpu-32/saved_model/')
    }



class ClazColor(object):
    """
    180 garish fully-saturated colors for display
    """
    def __init__(self,N):
        if N>180:
            raise ValueError("not enough colors for %d classes"%N)
        self.N=N
        self.cdict = {i: self._color(int(i*180/N)) for i in range(N)}
    
    def _color(self,hue):
        h = hue
        sat=255
        val=255
        color = map(int,np.squeeze(cv2.cvtColor(np.array([[[h,sat,val]]],
                                                         dtype=np.uint8),
                                              cv2.COLOR_HSV2BGR)
                                 )
        )
        color=tuple(color)
        return color
    def get(self,ref):
        if ref>=self.N:
            raise ValueError(ref)
        return self.cdict[ref]
    
    def get_map(self):
        l = [self.cdict[i] for i in range(self.N)]
        return np.array(l,dtype=np.uint8)


coco_colors = ClazColor(len(coco))

ade20k_colors = ClazColor(len(ade20k_to_coco))

pascal_to_coco={pascal.index(p):coco.index(p) for p in pascal if p in coco}




def annotate(img,drd,score_thresh=0.4,mode='coco',copy=False):
    if copy:
        img = img.copy()
    if mode=='coco':
        boxes,classes,scores = [np.squeeze(drd["detection_"+i])
                                for i in "boxes classes scores".split()]
    elif mode=='openimages':
        boxes,classes,scores = [np.squeeze(drd["detection_"+i])
                            for i in "boxes class_entities scores".split()]
    height,width = img.shape[0:2]
    #print(boxes.shape,classes.shape,scores.shape)
    for box,claz,score in zip(boxes,classes,scores):
        color = coco_colors.get(int(claz))
        ymin, xmin, ymax, xmax = box
        ymin=int(ymin*height)
        ymax=int(ymax*height)
        xmin=int(xmin*width)
        xmax=int(xmax*width)
        if score >= score_thresh:
            cv2.line(img,(xmin,ymin),(xmax,ymin),color)
            cv2.line(img,(xmax,ymin),(xmax,ymax),color)
            cv2.line(img,(xmax,ymax),(xmin,ymax),color)
            cv2.line(img,(xmin,ymax),(xmin,ymin),color)
            if mode=='coco':
                name = coco[int(claz)]
            elif mode=='openimages':
                name = claz
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,name,(xmin,ymin), font, 0.5,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(img,name,(xmin,ymin), font, 0.5,color,1,cv2.LINE_AA)
    return img



def annotate_sseg(img,segout,copy=False,mode='coco'):
    if type(segout) is not np.ndarray:
        segout = segout.numpy()
    if len(segout.shape) == 3:
        assert segout.shape[0]==1,"batch not supported"
        segout = segout[0,...]
    oot = np.zeros(segout.shape+(3,),np.uint8)
    if mode == 'coco':
        cmap = coco_colors.get_map()
    elif mode == 'ade20k':
        cmap = ade20k_colors.get_map()
    elif mode == 'pascal_to_coco':
        cmap = coco_colors.get_map()
        adjusted = [cmap[pascal_to_coco[i],:] for i in range(len(pascal))]
        cmap = np.uint8(adjusted)
    elif mode == 'ade20k_to_coco':
        cmap = coco_colors.get_map()
        adjusted = [cmap[ade20k_to_coco[i],:]
                    for i in range(len(ade20k_to_coco))]
        cmap = np.uint8(adjusted)
    else:
        raise ValueError(mode)
    for u in np.unique(segout):
        if u == 0 :continue
        oot[segout==u,:]=cmap[u,:]
    overlay = cv2.resize(oot,img.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)
    if copy:
        img = img.copy()
    img[...] = (0.5*img+0.5*overlay).astype(np.uint8)
    return img

def get_model(shortname):
    import models
    return models.load(shortname)

def getiou(boxA,boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def globalize_boxes(boxes,embedding,global_dims):
    """
    boxes: ratio of embedding w/h
    out: as ratio of global dims
    """
    assert len(boxes.shape)==2 and boxes.shape[1]==4,boxes.shape
    min_kp = boxes[:,0:2]
    max_kp = boxes[:,2:4]
    min_kp = globalize_points(min_kp,embedding,global_dims)
    max_kp = globalize_points(max_kp,embedding,global_dims)
    return np.concatenate([min_kp,max_kp],axis=1)

def globalize_keypoints(kp,embedding,global_dims):
    assert len(kp.shape)==3 and kp.shape[2]==2,kp.shape
    s = kp.shape
    p = globalize_points(kp.reshape((-1,2)),embedding,global_dims)
    return p.reshape(s)

def globalize_points(kp,embedding,global_dims):
    """
    given an embedding (box) inside of larger box of shape global_dims,
    convert each Y,X point in kp from normalized coordinates in the embedding
    to normalized coordinates in the global frame
    """
    assert len(kp.shape)==2 and kp.shape[1]==2,kp.shape
    kp = kp.copy()
    glob_h,glob_w = global_dims
    ymin,xmin,ymax,xmax = embedding
    assert 0 <= ymin <= glob_h
    assert 0 <= xmin <= glob_w 
    e_h = ymax-ymin
    e_w = xmax-xmin
    #separate X and Y:
    Y = kp[:,0]
    X = kp[:,1]
    #convert to local pixel coords:
    Y *= e_h
    X *= e_w
    #convert to global pixel coords:
    Y += ymin
    X += xmin
    #convert to global normalized coords:
    Y /= glob_h
    X /= glob_w
    #glue back together
    out =  np.stack([Y,X],axis=1)
    assert np.min(out) >= 0
    assert np.max(out) <= 1
    return out

def contains(outer_box,inner_box):
    yo,xo,Yo,Xo = outer_box
    yi,xi,Yi,Xi = inner_box

    return yi >= yo and xi >= xo and \
        Yi <= Yo and Xi <= Xo

@tf.function
def to_coco_labels(lmap,mode='pascal'):
    if mode=='pascal':
        xlator= pascal_to_coco
    if mode=='ade20k':
        import ade20k
        xlator= ade20k.ade20k_to_coco()
    intermediate = []
    for i in range(1,len(xlator)):
        intermediate.append(tf.where(lmap==i,xlator[i],0))
    #since there's no overlap, we can combine all maps with add
    #this allows for parallelism when above loop is traced.
    return tf.math.reduce_sum(intermediate,axis=0)

@tf.function
def crop_square_box(img_like, box, margin = 0.2, out_size = 513,
                    output_h_to_w_ratio = 1.0,
                    interp=tf.image.ResizeMethod.BILINEAR,
                    antialias=True):
    """
    try to get a square box with margin around box: a rectangular box.
    resize to side length out_size.
    if not possible, truncate box at image edge and resize
    return the resized image-like and its box coords.
    """

    if img_like.shape[0]==1:
        img_like = img_like[0,...]

    img_h = img_like.shape[0]
    img_w = img_like.shape[1]

    
    ymin_norm,xmin_norm,ymax_norm,xmax_norm = [box[i] for i in range(4)]

    ymin = ymin_norm*img_h
    ymax = ymax_norm*img_h
    xmin = xmin_norm*img_w
    xmax = xmax_norm*img_w
                   
    w = xmax-xmin
    h = ymax-ymin

    if output_h_to_w_ratio == 1: #so we can skip building extra graph in sq case
        longer = tf.math.reduce_max([w,h])
        h_comp = (longer - h)/2 + margin*longer
        w_comp = (longer - w)/2 + margin*longer
    else:
        #in the case where our input box is limited by width:
        w_comp_a = margin*w
        h_comp_a = margin*w*output_h_to_w_ratio +\
                   (w*output_h_to_w_ratio - h)/2
        #in the case where our input box is limited by height:
        w_comp_b = margin*h/output_h_to_w_ratio +\
                   (h/output_h_to_w_ratio - w)/2
        h_comp_b = margin*h
        #now have to tf.cond which one to use
        h_comp,w_comp = tf.cond(h/w > output_h_to_w_ratio,
                                lambda: [h_comp_b,w_comp_b],
                                lambda: [h_comp_a,w_comp_a])
        
        

    ymin -= h_comp
    ymax += h_comp
    xmin -= w_comp
    xmax += w_comp

    ymin = tf.reduce_max([ymin,0])
    ymax = tf.reduce_min([ymax,img_h])
    xmin = tf.reduce_max([xmin,0])
    xmax = tf.reduce_min([xmax,img_w])

    ymin = tf.cast(ymin,tf.int32)
    ymax = tf.cast(ymax,tf.int32)
    xmin = tf.cast(xmin,tf.int32)
    xmax = tf.cast(xmax,tf.int32)

    actual_box = tf.stack([ymin,xmin,ymax,xmax])
    
    cutout = img_like[ymin:ymax,xmin:xmax,...]

    resized = tf.image.resize(cutout,(out_size,int(out_size/output_h_to_w_ratio)),
                           method=interp,antialias=antialias)
    return tf.cast(resized,img_like.dtype), actual_box

@tf.function
def replace_box(whole,part,y,x):
    #idk how else we're supposed to substitute a box-shaped region within
    #another in tf. seems like it ought to be a single api call.
    vertical_merged = tf.concat([whole[0:y,x:x+part.shape[1],...],
                                 part,
                                 whole[y+part.shape[0]:,x:x+part.shape[1],...]],
                                axis=0)
    merged = tf.concat([whole[:,0:x,...],
                        vertical_merged,
                        whole[:,x+part.shape[1]:,...]],
                       axis=1)
    return merged

def seg_multiscale(seg_model, seg_side=513):

    def merge(whole,part,y,x):
        masked_region = whole[y:y+part.shape[0],
                              x:x+part.shape[1],
                              ...]
        masked_region = tf.where(masked_region==0,part,masked_region)
        return replace_box(whole,masked_region,y,x)
    def seg_in_region(img,ymin,xmin,ymax,xmax):
        roi = img[ymin:ymax,xmin:xmax,...]
        seg_in = tf.image.resize(roi,(seg_side,seg_side),antialias=True)
        seg_in = tf.cast(seg_in,tf.uint8)
        seg_in = tf.expand_dims(seg_in,axis=0)
        seg_out = seg_model(seg_in)
        seg_out = seg_out[0,...]
        if len(seg_out.shape)==2:
            munge_channels=True
            seg_out = tf.expand_dims(seg_out,axis=2)
        else:
            munge_channels=False
        seg_roi = tf.image.resize(seg_out,roi.shape[:2],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        if munge_channels:
            seg_roi = seg_roi[:,:,0]
        return seg_roi
    def overlapped_quads(height,width):
        hr = [int(height*i/4) for i in range(5)]
        wr = [int(width*i/4) for i in range(5)]
        boxen = []
        for j in range(3):
            for i in range(3):
                boxen.append((hr[j],wr[i],hr[j+2],wr[i+2]))
        return boxen
    def overlapped_boxen(height,width):
        hr = [int(height*i/4) for i in range(5)]
        wr = [int(width*i/6) for i in range(7)]
        boxen = []
        for j in range(3):
            for i in range(5):
                boxen.append((hr[j],wr[i],hr[j+2],wr[i+2]))
        return boxen
    def quadrants(height,width):
        hm = int(height/2)
        wm = int(width/2)
        return [
            (0,0,hm,wm),
            (0,wm,hm,width),
            (hm,wm,height,width),
            (hm,0,height,wm)
            ]
    def get_subregions(height,width):
        return overlapped_boxen(height,width)
    @tf.function
    def multiscale(img):
        height = img.shape[0]
        width  = img.shape[1]
        seg_whole = seg_in_region(img,0,0,height,width)
        chops = get_subregions(height,width)
        for (ymin,xmin,ymax,xmax) in chops:
            segmented = seg_in_region(img,ymin,xmin,ymax,xmax)
            seg_whole = merge(seg_whole,segmented,ymin,xmin)
        return seg_whole
    return multiscale
        
    
    

#####################################################################################
#### stuff below is from failed attempt to make this run anywhere near framerate ####
#####################################################################################

def load_batched_detector(name,batch_size,image_w):
    bolus = tf.saved_model.load(os.path.join(MODELS_DIR,name))
    s = tf.constant(np.array([[image_w,image_w,3]]*batch_size,
                             dtype=np.int32))
    @tf.function
    def detect(image_batch_tensor):
        #resize to desired:
        images = tf.image.resize(image_batch_tensor,(image_w,image_w),
                                 method=tf.image.ResizeMethod.BILINEAR,
                                 antialias=True) / 255 - 0.5
        boxes,classes,scores = bolus.detect_fn(images,s)
        classes = tf.cast(tf.math.round(classes),tf.int32)+1#idk why off by one :?
        detected = [boxes,classes,scores]
        return {'detection_'+k:d for (k,d) in zip('boxes classes scores'.split(),
                                                  detected)}
    return detect

class BatchProcessor(object):
    def __init__(self,tf_func,batch_size,dtype=np.float32,out_notify=None):
        self.alive=True
        self.dtype=dtype
        self.tf_func=tf_func
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self._q = []
        self._out_q = []
        self.output_ready = threading.Event()
        self.service_thread = threading.Thread(target=self.run)
        self.out_notify = out_notify
        atexit.register(self.kill)
        self.start_time = time.time()
        self.frame_count=0
        self.sys_excepthook = sys.excepthook
        sys.excepthook = self.excepthook
        self.service_thread.setDaemon(True)#atexit joins first :(
        self.service_thread.start()

    @property
    def framerate(self):
        return self.frame_count/(time.time()-self.start_time)

    def run(self):
        while self.alive:
            work = []
            with self.lock:
                while len(self._q) < self.batch_size:
                    if not self.alive: return
                    self.cond.wait()
                    if not self.alive: return
                while len(self._q) >= self.batch_size:
                    chunk = self._q[:self.batch_size]
                    chunk = np.array(chunk,dtype=self.dtype)
                    work.append(chunk)
                    del self._q[:self.batch_size]
            for work_item in work:
                if not self.alive: return
                out = self.tf_func(work_item)
                self.frame_count+=self.batch_size
                with self.lock:
                    self._out_q.append(out)
                self.output_ready.set()
                if self.out_notify:
                    with self.out_notify:
                        self.out_notify.notify()
    def kill(self):
        if self.alive:
            logging.info("kill BatchProcessor")
        with self.lock:
            self.alive=False
            self.cond.notify()
    def has_output(self):
        with self.lock:
            return len(self._out_q)>0
    def pop(self):
        with self.lock:
            return self._out_q.pop(0)
    def push(self,frame):
        with self.lock:
            self._q.append(frame)
            self.cond.notify()
    def extend(self,frames):
        with self.lock:
            self._q.extend(frames)
            self.cond.notify()
    def excepthook(self,type,value,traceback):
        self.kill()
        self.sys_excepthook(type,value,traceback)


class BatchObjectDetector(BatchProcessor):
    def __init__(self,model_name=ADK_SSD,batch_size=16,input_img_shape=(720,1280,3),
                 image_w=300,dtype=np.float32,**kwargs):
        model = load_batched_detector(model_name,batch_size,image_w)
        self.input_image_shape = (batch_size,)+input_img_shape
        logging.info("warming up detector")
        model(np.zeros(self.input_image_shape,dtype))
        super().__init__(model,batch_size=batch_size,dtype=dtype,**kwargs)


class Gate(object):
    def __init__(self,processors):
        self.processors = processors
        self.out_queues = [[] for p in self.processors]
        self.frame_queue = []
        self.frame_count = 0
        self.batch_size = processors[0].batch_size
        if any(p.batch_size!=self.batch_size for p in processors):
            raise ValueError("incompatible batch sizes")
        
    def push(self,frame):
        self.frame_count+=1
        self.frame_queue.append(frame)
        for processor in self.processors:
            processor.push(frame)
        if any(len(q)==0 for q in self.out_queues):
            if self.frame_count > 2*self.batch_size:
                #require all output to be ready now
                for processor,q in zip(self.processors,self.out_queues):
                    while not processor.has_output():
                        processor.output_ready.wait(0.1)
        for processor,q in zip(self.processors,self.out_queues):
            if processor.has_output():
                out = processor.pop()
                out = self.convert(out)
                q.extend(out)
        if self.frame_count > 2*self.batch_size:
            return self.frame_queue.pop(0),[q.pop(0) for q in self.out_queues]
        else:
            return None
    def convert(self,out):
        if type(out) is tf.python.framework.ops.EagerTensor:
            return out.numpy()
        if type(out) is np.ndarray:
            return out
        elif type(out) is dict:
            keys = list(out.keys())
            l = len(out[keys[0]])
            converted_1l = {k:self.convert(out[k]) for k in keys}
            return [{k:converted_1l[k][i] for k in keys} for i in range(l)]
        else:
            raise ValueError(type(out))

# def _load(url,style='v1'):
#     logging.info("loading %r"%url)
#     detector = hub.load(url)
#     logging.info("done loading")
#     if style == 'v1':
#         return detector.signatures['default']
#     elif style =='v2':
#         return hub.KerasLayer(detector)
#     else:
#         raise ValueError("bad style %r"%style)

# def load(fname,style='v2'):
#     if style=='v2':
#         m = tf.saved_model.load(fname)
#         logging.info(m.signatures)
#         return m
#     elif style='adk'
#     else:
#         raise ValueError(style)



# def _replicate(model_url,style='v1',n_replicas=2): #no speedup
#     detector = load(model_url,style=style)
#     #most pretrained models don't support batched inputs.
#     #even if they did, we want to be able to process
#     #images of different sizes simultaneously, and the
#     #model's resize machinery would probs not interact
#     #well with ragged tensor input
#     @tf.function#todo ignore shapes !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     def replicated(*args):
#         #each argument is an image, may be different sizes
#         #cannot pass a list or else tf.function would fail.
#         return [detector(arg) for arg in args]
#     return replicated

# def _replicate(model_url,style='v1',n_replicas=2): #no speedup
#     detectors = [load(model_url,style=style) for i in range(n_replicas)]
#     logging.info("loaded %d detector copies"%n_replicas)
#     @tf.function
#     def replicated(*args):
#         return [detector(arg) for arg,detector in zip(args,detectors)]
#     return replicated

# def _replicate(model_url,style='v1',n_replicas=2): #no speedup
#     detector = load(model_url,style=style)
#     inputs = [tf.keras.Input(shape=(None,None,3),batch_size=1,dtype=tf.uint8)
#               for i in range(n_replicas)]
#     outputs = [detector(inp,training=False) for inp in inputs]
#     return tf.keras.Model(inputs=inputs,outputs=outputs)

# def _replicate(model_url,style='v1',n_replicas=2): #blows out
#     detectors = []
#     for i in range(n_replicas):
#         g = tf.Graph()
#         with g.as_default():
#             detectors.append(load(model_url,style=style))
#             detectors[-1](np.ones((1,720,1280,3),np.uint8))
#     @tf.function
#     def replicated(*args):
#         return [detector(arg) for arg,detector in zip(args,detectors)]
#     return replicated


# def replicate(model_url,style='v1',n_replicas = 2):#blows out as expected
#     detector = load(model_url,style=style)
#     inputs = tf.keras.Input(shape=(None,None,3),batch_size=n_replicas,dtype=tf.uint8)#hbd
#     outputs= detector(inputs)
#     return tf.keras.Model(inputs=inputs,outputs=outputs)

# def _replicate(model_url,style='v1',n_replicas = 2): #no speedup
#     detector = load(model_url,style=style)
#     inputs = tf.keras.Input(shape=(None,None,3),batch_size=n_replicas,dtype=tf.uint8)#hbd
#     _inputs = tf.split(inputs,num_or_size_splits=n_replicas,axis=0)
#     outputs = [detector(i) for i in _inputs]
#     return tf.keras.Model(inputs=inputs,outputs=outputs)
    
    








def benchmark_single_model():
    #detector = load('./models/faster_rcnn_openimages_v4_inception_resnet_v2/')
    detector = load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",style='v2')
    image = (np.random.random((1,720,1280,3))*255).astype(np.uint8)
    #image = tf.constant(image)
    detector(image)
    import timeit
    print(timeit.timeit(lambda:detector(image),number=100))

def benchmark_replicate_model():
    #detector = replicate('./models/faster_rcnn_openimages_v4_inception_resnet_v2/',
    #                     n_replicas=4)
    detector = replicate(
        "./models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/checkpoint",
                         n_replicas=4,style='v2')
    #image = (np.random.random((1,720,1280,3))*255).astype(np.uint8)
    image = (np.random.random((4,720,1280,3))*255).astype(np.uint8)
    #image = tf.constant(image)
    #detector(*([image]*4))
    detector(image)
    import timeit
    #print(timeit.timeit(lambda:detector(*([image]*4)),number=100))
    print(timeit.timeit(lambda:detector(image),number=100))

if __name__=='__main__':
    benchmark_replicate_model()

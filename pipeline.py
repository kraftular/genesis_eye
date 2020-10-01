from detector_foo import get_model, annotate, annotate_sseg, to_coco_labels, crop_square_box, seg_multiscale
import detector_foo #todo refactor above !!!!
import human_foo
from models import load
import vproc
import logging
import cv2
import pyzed.sl as sl
from tqdm import tqdm
import numpy as np
import gc
from coco import coco


DEBUG=False

if DEBUG:
    logging.basicConfig(level=logging.DEBUG)

def imshow(name,img):
    if DEBUG:
        cv2.imshow(name,img)
        cv2.waitKey(1)
def destroy_windows():
    if DEBUG:
        cv2.destroyAllWindows()


class DetectionCoCo(object):
    def __init__(self,detector_name='coco_centernet'):
        self.detector_name = detector_name
        self._model = None

    @property
    def model(self):
        if self._model:
            return self._model
        else:
            self._model = get_model(self.detector_name)
            return self._model

    def process_svo(self,svo):
        h5file = vproc.get_hdf5_for_svo(svo)
        vproc.process_from_h5file(h5file,
                                  vproc.h5_extractor(self.processor)
        )

    def detect(self,image):
        out = self.model(image[np.newaxis,...])
        return {k:out[k].numpy() for k in out}

    def processor(self,left_image,right_image):
        detect = self.detect(left_image)
        output_l = {'left_'+k:detect[k] for k in detect}
        detect = self.detect(right_image)
        output_r = {'right_'+k:detect[k] for k in detect}
        imshow("boxen",annotate(right_image,detect,copy=False))
        return {**output_l,**output_r}
    def __call__(self,svo):
        self.process_svo(svo)

class RefineDetections(DetectionCoCo):
    """
    refine object detections nearby to humans
    """
    human_threshold = 0.2
    max_humans = 10
    iou_threshold = 0.4 #for determining if two boxes are (not) the same
    skel_pt_thresh = 15 #pixel dist between joints to consider same
    def processor(self,
                  left_image,
                  left_detection_boxes,
                  left_detection_classes,
                  left_detection_scores,
                  left_detection_keypoints,
                  left_detection_keypoint_scores,
                  
                  right_image,
                  right_detection_boxes,
                  right_detection_classes,
                  right_detection_scores,
                  right_detection_keypoints,
                  right_detection_keypoint_scores):
        
        detect = self.detect_refined(left_image,
                                     left_detection_boxes,
                                     left_detection_classes,
                                     left_detection_scores,
                                     left_detection_keypoints,
                                     left_detection_keypoint_scores,vis=True)
        output_l = {'left_refined_'+k:detect[k] for k in detect}
        box_disp = annotate(left_image,detect,copy=True)
        for (k,s) in zip(output_l['left_refined_detection_keypoints'][0],
                         output_l['left_refined_detection_keypoint_scores'][0]):
            human_disp = human_foo.draw_human(box_disp,k,s)
                                
        imshow("current left frame",human_disp)
        detect = self.detect_refined(right_image,
                                     right_detection_boxes,
                                     right_detection_classes,
                                     right_detection_scores,
                                     right_detection_keypoints,
                                     right_detection_keypoint_scores)
        output_r = {'right_refined_'+k:detect[k] for k in detect}
        return {**output_l,**output_r}

    def detect_refined(self,image,boxes,classes,scores,kps,kp_scores,vis=False):
        boxes,classes,scores,kps,kp_scores = (np.squeeze(thing) for thing in
                                            (boxes,classes,scores,kps,kp_scores))
        refined_bcs = [] #refined dicts
        for (box,claz,score,kp,kp_score) in zip(boxes,classes,scores,kps,kp_scores):
            if claz==coco.index('person') and score >= self.human_threshold:
                #print("detecting!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")
                window,actual_box = crop_square_box(image,box,out_size=1024)
                window = window.numpy()
                actual_box = actual_box.numpy()
                #imshow("window",window)
                detected = self.detect(window)
                #destroy batch dimensions, to match others!
                detected = {k:np.squeeze(detected[k]) for k in detected}
                detected['detection_boxes'] = detector_foo.globalize_boxes(
                    detected['detection_boxes'],
                    actual_box,
                    (image.shape[0],image.shape[1]))
                detected['detection_keypoints'] = detector_foo.globalize_keypoints(
                    detected['detection_keypoints'],
                    actual_box,
                    (image.shape[0],image.shape[1]))
                refined_bcs.append([detected,actual_box])
        if len(refined_bcs)>self.max_humans:
            raise ValueError("too many human detections! %d"%len(refined_bcs))
        #now, refined bcs contains dicts that are either additional detections,
        #not found in the original whole image, or refined (closeup) versions of
        #existing detections. we have to reconcile them with the existing detections.
        out = dict(detection_boxes=[],
                   detection_classes=[],
                   detection_scores=[],
                   detection_keypoints = [],
                   detection_keypoint_scores=[])
        #XXX this ignores detections not seen at all in global dets. if they were
        #detected but their scores are just weak, it still might be ok.
        for (box,claz,score,kp,kp_score) in zip(boxes,classes,scores,kps,kp_scores):
            for d,actual_box in refined_bcs:
                rboxes = np.squeeze(d['detection_boxes'])
                rclasses = np.squeeze(d['detection_classes'])
                rscores = np.squeeze(d['detection_scores'])
                for (rbox,rclaz,rscore) in zip(rboxes,rclasses,rscores):
                    if rclaz != claz: continue
                    if detector_foo.getiou(rbox,box) > self.iou_threshold and\
                       rscore > score:
                        box = rbox
                        score = rscore
                    #commented, above, checks that the newly-found box is about
                    #the same as the full-frame box. but that limits our refined
                    #boxen to be the same as our coarse boxen. what we really want
                    #is to fill the actual_box with as many higher scoring boxen
                    #as we can fit, under the constraint that we have to throw out
                    #an old box to get a new one must be fixed number
                    #if detector_foo.contains(box,rbox) and rscore > score and \
                    #   detector_foo.getiou(rbox,box) < self.iou_threshold:
                    #    #the new box is different than the old one, and higher
                    #    #scoring... but this runs the risk of greedily replacing
                    #    #unrelated box with higher scoring box. does guarantee that
                    #    #refinement box contains highest scoring refined box unless
                    #    #it's the same as an existing one...
                    #    box = rbox
                    #    score = rscore
            out['detection_boxes'].append(box)
            out['detection_classes'].append(claz)
            out['detection_scores'].append(score)
        #finally, for the detections that are people, figure out which skeleton
        #is closest in "skel threshold distance"
        for (claz,score,kp,kp_score) in zip(classes,scores,kps,kp_scores):
            if claz != coco.index('person') or score < self.human_threshold:
                out['detection_keypoints'].append(kp)#XXX wrong, but we don't care
                out['detection_keypoint_scores'].append(kp_score)#XXX this is wrong
                #the above uses the wrong values for keypoints of non-people. but
                #who cares, because we only need keypoints for people.
            else:#is a person
                distance = 0.5 #the old box is better than severely mismatched box
                best_skel = None
                best_score = None
                for d,actual_box in refined_bcs:
                    rkps = np.squeeze(d['detection_keypoints'])
                    rkpscores = np.squeeze(d['detection_keypoint_scores'])
                    rclasses = np.squeeze(d['detection_classes'])
                    for (rkp,rkp_score,rclaz) in zip(rkps,rkpscores,rclasses):
                        if rclaz != claz: continue
                        #both person detections
                        d_star = human_foo.skel_dist(self.skel_pt_thresh,
                                                     image.shape[0],image.shape[1],
                                                     kp,kp_score,
                                                     rkp,rkp_score)
                        if d_star < distance:
                            distance = d_star
                            best_skel = rkp
                            best_score = rkp_score
                if best_skel is not None:
                    out['detection_keypoints'].append(best_skel)
                    out['detection_keypoint_scores'].append(best_score)
                    #print ("using refined!!!!!!!!!!!!!")
                else:
                    out['detection_keypoints'].append(kp)
                    out['detection_keypoint_scores'].append(kp_score)
                    #print("using old")
        out = {k:np.array(out[k]) for k in out}
        assert all(len(out[k])==len(boxes) for k in out)
        #restore the batch dimension, to match other data:
        out = {k:out[k][np.newaxis,...] for k in out}
        return out
                    
                

class GlobalSegmentation(object):
    def __init__(self):
        self.model =  seg_multiscale(load('deeplabv3_pascal_trainval'))

    def process_svo(self,svo):
        h5file = vproc.get_hdf5_for_svo(svo)
        vproc.process_from_h5file(h5file,
                                  vproc.h5_extractor(self.processor))

    def segment(self,image):
        out = self.model(image)
        out = to_coco_labels(out,mode='pascal')
        out = out.numpy()
        return {'global_seg':out}

    def processor(self,left_image,right_image):
        segmented = self.segment(left_image)
        output_l = {'left_'+k:segmented[k] for k in segmented}
        segmented = self.segment(right_image)
        output_r = {'right_'+k:segmented[k] for k in segmented}
        imshow("seg",annotate_sseg(right_image,segmented['global_seg'],
                                   copy=True,mode='coco'))
        return {**output_l,**output_r}

    def __call__(self,svo):
        self.process_svo(svo)
        

class StoreCameraConfig(object):
    def process_svo(self,svo):
        from zed_geometry import extract_params
        path = vproc.get_svo_path(svo)
        camera_info = extract_params(path)
        h5file = vproc.get_hdf5_for_svo(svo)
        vproc.store_array_object('camera_info',camera_info,h5file)
    def __call__(self,svo):
        self.process_svo(svo)
        

def extract_zed_images(svo_files):
    def extractor(zed):
        mat = sl.Mat()
        runtime = sl.RuntimeParameters()
        while True:
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(mat,sl.VIEW.LEFT)
                imgl = mat.get_data()[...,:3].copy()
                imshow("left",imgl)
                zed.retrieve_image(mat,sl.VIEW.RIGHT)
                imgr = mat.get_data()[...,:3].copy()
                yield dict(left_image=imgl.astype(np.uint8),
                           right_image=imgr.astype(np.uint8))
            else:
                break
    for svo_file in svo_files:
        path = vproc.get_svo_path(svo_file)
        h5file = vproc.get_hdf5_for_svo(svo_file)
        vproc.process_from_svo(path,extractor,h5file)

class SVOExtractImages(object):
    def __call__(self,svo):
        extract_zed_images([svo])



        

def pipeline(stages,svo_filenames):
    #this doesn't work. the problem is that it's impossible to clean up
    #all of tf and zed's internal memory leaks, and models start OOM problems
    #after several iterations. gonna have to spin up subprocesses.
    import tensorflow as tf
    for stage_class in stages:
        stage = stage_class()#create an instance, which may build some tf graph
        logging.info("processing stage: %s"%stage_class.__name__)
        for svo_filename in tqdm(svo_filenames):
            logging.info("processing file: %s"%svo_filename)
            stage(svo_filename)
        destroy_windows()
        del stage
        gc.collect()#force deletion of graph linked to stage object
        tf.keras.backend.clear_session()#this is mostly superstition
        


              
            
                                                                      
#####################################################################################
# below, just debugging stuff and other nonsense                                    #
#####################################################################################

        
def playback_test():
    svo = 'HD720_SN27165053_16-29-04.svo'
    h5file = vproc.get_hdf5_for_svo(svo)
    vproc.print_channels(h5file)
    def processor(left_detection_boxes,
                  left_detection_classes,
                  left_detection_scores,
                  left_image):
        d = dict(detection_boxes=left_detection_boxes,
                 detection_scores=left_detection_scores,
                 detection_classes=left_detection_classes)
        img = annotate(left_image,d,score_thresh=0.3,copy=True)
        imshow("detections",img)
    vproc.playback_h5file(h5file,processor)

def playback_test2():
    svo = 'HD720_SN27165053_16-29-04.svo'
    h5file = vproc.get_hdf5_for_svo(svo)
    vproc.print_channels(h5file)
    from models import load
    import tensorflow as tf
    m = load('deeplabv3_pascal_trainval')
    #m = load('deeplabv3_xception_ade20k_train')
    def processor(left_image,
                  left_detection_boxes,
                  left_detection_classes,
                  left_detection_scores):
        left_detection_boxes = np.squeeze(left_detection_boxes)
        left_detection_scores= np.squeeze(left_detection_scores)
        left_detection_classes = np.squeeze(left_detection_classes)
        height,width = left_image.shape[0:2]
        composite = np.zeros((height,width),np.int32)
        
        for box,score,claz in zip(left_detection_boxes,
                                  left_detection_scores,
                                  left_detection_classes):
            if score < 0.3: continue
            #ymin, xmin, ymax, xmax = box
            #ymin=int(ymin*height)
            #ymax=int(ymax*height)
            #xmin=int(xmin*width)
            #xmax=int(xmax*width)
            #det = left_image[ymin:ymax,xmin:xmax,:]
            #img = cv2.resize(det,(513,513))#**

            img,actual_box = crop_square_box(left_image, box)
            ymin, xmin, ymax, xmax = actual_box.numpy()
            #ymin=int(ymin)
            #ymax=int(ymax)
            #xmin=int(xmin)
            #xmax=int(xmax)
            
            img = img[np.newaxis,...]
            sseg = m(tf.constant(img))
            sseg = to_coco_labels(sseg,mode='pascal')
            sseg = sseg.numpy()[0,...].astype(np.int32)
            
            sseg[sseg!=claz]=0
            sseg = cv2.resize(sseg,(xmax-xmin,ymax-ymin),
                              interpolation=cv2.INTER_NEAREST)
            window = composite[ymin:ymax,xmin:xmax]
            #print("numpy",window.shape,sseg.shape)
            window[window==0]=sseg[window==0]
            #window[...]=sseg

        disp = annotate_sseg(left_image,composite,copy=True,mode='coco')
        d = dict(detection_boxes=left_detection_boxes,
                 detection_scores=left_detection_scores,
                 detection_classes=left_detection_classes)
        disp = annotate(disp,d,score_thresh=0.3)
        imshow('seg',disp)
            

            
            
        
    vproc.playback_h5file(h5file,processor,start=1000)


def playback_test3():
    #conclusion: ade20k is too noisy or else the labels are inscrutable
    svo = 'HD720_SN27165053_16-29-04.svo'
    h5file = vproc.get_hdf5_for_svo(svo)
    vproc.print_channels(h5file)
    from models import load
    import tensorflow as tf
    m = load('deeplabv3_xception_ade20k_train')
    def processor(left_image,
                  left_detection_boxes,
                  left_detection_classes,
                  left_detection_scores):
        left_detection_boxes = np.squeeze(left_detection_boxes)
        left_detection_scores= np.squeeze(left_detection_scores)
        left_detection_classes = np.squeeze(left_detection_classes)
        height,width = left_image.shape[0:2]
        #composite = np.zeros((height,width),np.int32)
        
        sseg = m(tf.constant(cv2.resize(left_image,(513,513))[np.newaxis,...]))
        sseg = sseg.numpy()[0,...].astype(np.int32)
        sseg = cv2.resize(sseg,(width,height),
                              interpolation=cv2.INTER_NEAREST)
            
        disp = annotate_sseg(left_image,sseg,copy=True,mode='ade20k')
        d = dict(detection_boxes=left_detection_boxes,
                 detection_scores=left_detection_scores,
                 detection_classes=left_detection_classes)
        disp = annotate(disp,d,score_thresh=0.3)
        imshow('seg',disp)
            

            
            
        
    vproc.playback_h5file(h5file,processor,start=1000)


def playback_test4():
    #conclusion: ade20k is too noisy or else the labels are inscrutable
    svo = 'HD720_SN27165053_16-29-04.svo'
    h5file = vproc.get_hdf5_for_svo(svo)
    vproc.print_channels(h5file)
    from models import load
    import tensorflow as tf
    m = load('deeplabv3_pascal_trainval')
    from detector_foo import seg_multiscale
    m = seg_multiscale(m)
    def processor(left_image,
                  left_detection_boxes,
                  left_detection_classes,
                  left_detection_scores):
        left_detection_boxes = np.squeeze(left_detection_boxes)
        left_detection_scores= np.squeeze(left_detection_scores)
        left_detection_classes = np.squeeze(left_detection_classes)
        height,width = left_image.shape[0:2]
        #composite = np.zeros((height,width),np.int32)
        
        sseg = m(tf.constant(left_image))

        sseg = to_coco_labels(sseg,mode='pascal')
        
            
        disp = annotate_sseg(left_image,sseg,copy=True,mode='coco')
        d = dict(detection_boxes=left_detection_boxes,
                 detection_scores=left_detection_scores,
                 detection_classes=left_detection_classes)
        disp = annotate(disp,d,score_thresh=0.3)
        imshow('seg',disp)
    vproc.playback_h5file(h5file,processor,start=1000)

def playback_test5():
    svo = 'HD720_SN27165053_16-29-04.svo'
    h5file = vproc.get_hdf5_for_svo(svo)
    vproc.print_channels(h5file)
    from models import load
    import tensorflow as tf
    def processor(left_image,
                  left_detection_boxes,
                  left_detection_classes,
                  left_detection_scores,
                  left_global_seg):
        left_detection_boxes = np.squeeze(left_detection_boxes)
        left_detection_scores= np.squeeze(left_detection_scores)
        left_detection_classes = np.squeeze(left_detection_classes)
        height,width = left_image.shape[0:2]
        #composite = np.zeros((height,width),np.int32)
    

        sseg = left_global_seg
        
            
        disp = annotate_sseg(left_image,sseg,copy=True,mode='coco')
        d = dict(detection_boxes=left_detection_boxes,
                 detection_scores=left_detection_scores,
                 detection_classes=left_detection_classes)
        disp = annotate(disp,d,score_thresh=0.3)
        imshow('seg',disp)
    vproc.playback_h5file(h5file,processor)


def playback_test6():
    #svo = 'HD720_SN27165053_16-29-04.svo'
    #svo='hug-1.svo'
    #svo = 'depressed-drink-1.svo'
    #svo = 'dance-headbutt-1.svo'
    #svo = 'read-take-1.svo'
    svo = 'wave-1.svo'
    h5file = vproc.get_hdf5_for_svo(svo)
    vproc.print_channels(h5file)
    from models import load
    import tensorflow as tf
    from tf2_pose.estimator import TfPoseEstimator
    from tf2_pose.networks import get_graph_path
    est =  TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
    use_est=False
    def processor(left_image,
                  left_refined_detection_boxes,
                  left_refined_detection_classes,
                  left_refined_detection_scores,
                  left_refined_detection_keypoints,
                  left_refined_detection_keypoint_scores):

        left_refined_detection_keypoint_scores =human_foo.remove_duplicate_skeletons(
            left_refined_detection_classes,
            left_refined_detection_scores,
            left_refined_detection_keypoints,
            left_refined_detection_keypoint_scores)
        left_refined_detection_keypoint_scores =human_foo.remove_low_val_skeletons(
            left_refined_detection_classes,
            left_refined_detection_keypoint_scores)
        left_refined_detection_scores =human_foo.reset_person_detections(
            left_refined_detection_classes,
            left_refined_detection_scores,
            left_refined_detection_keypoint_scores)
        
        left_detection_boxes = np.squeeze(left_refined_detection_boxes)
        left_detection_scores= np.squeeze(left_refined_detection_scores)
        left_detection_classes = np.squeeze(left_refined_detection_classes)
        left_detection_keypoints = np.squeeze(left_refined_detection_keypoints)
        left_detection_keypoint_scores = np.squeeze(left_refined_detection_keypoint_scores)
        height,width = left_image.shape[0:2]
        disp = left_image.copy()
        #print(left_detection_keypoints.shape,left_detection_keypoint_scores.shape)
        for box,score,claz,kps,confs in zip(left_detection_boxes,
                                  left_detection_scores,
                                      left_detection_classes,
                                      left_detection_keypoints,
                                      left_detection_keypoint_scores):
            if score < 0.3: continue
            if claz != 1: continue

            if use_est:
                cutout,actual_box = crop_square_box(left_image,box,out_size=432,
                                                    output_h_to_w_ratio=432./368)
                cutout,actual_box = cutout.numpy(),actual_box.numpy()
                humans = est.inference(cutout, resize_to_default=True,
                                       upsample_size=4)
                y,x,Y,X = actual_box
                window = disp[y:Y,x:X,...]
                TfPoseEstimator.draw_humans(window,humans,imgcopy=False)
            else:
                
                human_foo.draw_human(disp,kps,confs)
            
        
        d = dict(detection_boxes=left_refined_detection_boxes,
                 detection_scores=left_refined_detection_scores,
                 detection_classes=left_refined_detection_classes)
        disp = annotate(disp,d,score_thresh=0.3)
        imshow('display',disp)
    vproc.playback_h5file(h5file,processor)

def playback_test6point5():
    #svo = 'HD720_SN27165053_16-29-04.svo'
    #svo='hug-1.svo'
    #svo = 'depressed-drink-1.svo'
    #svo = 'dance-headbutt-1.svo'
    #svo = 'wave-1.svo'
    svo = 'give-book-2.svo'
    #svo = 'read-take-1.svo'
    h5file = vproc.get_hdf5_for_svo(svo)
    vproc.print_channels(h5file)
    from models import load
    import tensorflow as tf
    from tf2_pose.estimator import TfPoseEstimator
    from tf2_pose.networks import get_graph_path
    st = human_foo.SkeletonTracker()
    def processor(left_image,
                  left_refined_detection_boxes,
                  left_refined_detection_classes,
                  left_refined_detection_scores,
                  left_refined_detection_keypoints,
                  left_refined_detection_keypoint_scores):

        detection_boxes = left_refined_detection_boxes
        detection_scores= left_refined_detection_scores
        detection_classes = left_refined_detection_classes
        detection_keypoints = left_refined_detection_keypoints
        detection_keypoint_scores = left_refined_detection_keypoint_scores

        detection_keypoint_scores =human_foo.remove_duplicate_skeletons(
            detection_classes,
            detection_scores,
            detection_keypoints,
            detection_keypoint_scores)
        detection_keypoint_scores =human_foo.remove_low_val_skeletons(
            detection_classes,
            detection_keypoint_scores)
        detection_boxes,detection_scores =human_foo.reset_person_detections(
            detection_classes,
            detection_boxes,
            detection_scores,
            detection_keypoints,
            detection_keypoint_scores)

        #detection_keypoint_scores = human_foo.remove_skel_joints_outside_box(
        #    detection_keypoints,
        #    detection_keypoint_scores,
        #    detection_boxes)
        
        
        height,width = left_image.shape[0:2]
        disp = left_image.copy()
        #print(left_refined_detection_keypoints.shape,left_refined_detection_keypoint_scores.shape)
        
        st.update(detection_classes,
                  detection_boxes,
                  detection_scores,
                  detection_keypoints,
                  detection_keypoint_scores)

        human_foo.draw_tracked(disp,st.get_tracked())

            
        
        #d = dict(detection_boxes=detection_boxes,
        #         detection_scores=detection_scores,
        #         detection_classes=detection_classes)
        #disp = annotate(disp,d,score_thresh=0.3)
        imshow('display',disp)
    vproc.playback_h5file(h5file,processor)

def playback_test7():
    svo = 'HD720_SN27165053_16-29-04.svo'
    #svo='hug-1.svo'
    #svo = 'depressed-drink-1.svo'
    #svo = 'dance-headbutt-1.svo'
    #svo = 'wave-1.svo'
    #svo = 'give-book-2.svo'
    #svo = 'read-take-1.svo'
    h5file = vproc.get_hdf5_for_svo(svo)
    vproc.print_channels(h5file)
    from models import load
    import tensorflow as tf
    from tf2_pose.estimator import TfPoseEstimator
    from tf2_pose.networks import get_graph_path
    st = human_foo.SkelTracker3D()
    def processor(left_image,
                  left_refined_detection_boxes,
                  left_refined_detection_classes,
                  left_refined_detection_scores,
                  left_refined_detection_keypoints,
                  left_refined_detection_keypoint_scores,
                  
                  right_image,
                  right_refined_detection_boxes,
                  right_refined_detection_classes,
                  right_refined_detection_scores,
                  right_refined_detection_keypoints,
                  right_refined_detection_keypoint_scores,

                  camera_info
                  
    ):

        process(left_image,

                (left_refined_detection_classes,
                 left_refined_detection_boxes,
                 left_refined_detection_scores,
                 left_refined_detection_keypoints,
                 left_refined_detection_keypoint_scores),
                  
                (right_refined_detection_classes,
                 right_refined_detection_boxes,
                 right_refined_detection_scores,
                 right_refined_detection_keypoints,
                 right_refined_detection_keypoint_scores),

                camera_info
                )
    def process(img,left_args,right_args,camera_info):
        
        st.update(left_args,right_args)
        
        imshow('frame',img)
    vproc.playback_h5file(h5file,processor,array_args=['camera_info'])

def playback_test_overlay():
    #svo = 'HD720_SN27165053_16-29-04.svo'
    #svo='hug-1.svo'
    #svo = 'depressed-drink-1.svo'
    svo = 'dance-headbutt-1.svo'
    #svo = 'wave-1.svo'
    #svo = 'read_take-1.svo'
    h5file = vproc.get_hdf5_for_svo(svo)
    vproc.print_channels(h5file)
    from models import load
    import tensorflow as tf
    from tf2_pose.estimator import TfPoseEstimator
    from tf2_pose.networks import get_graph_path
    est =  TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
    use_est=False
    def processor(left_image,
                  left_detection_boxes,
                  left_detection_classes,
                  left_detection_scores,
                  left_detection_keypoints,
                  left_detection_keypoint_scores,
                  right_image,
                  right_detection_boxes,
                  right_detection_classes,
                  right_detection_scores,
                  right_detection_keypoints,
                  right_detection_keypoint_scores):
        left_detection_boxes = np.squeeze(left_detection_boxes)
        left_detection_scores= np.squeeze(left_detection_scores)
        left_detection_classes = np.squeeze(left_detection_classes)
        left_detection_keypoints = np.squeeze(left_detection_keypoints)
        left_detection_keypoint_scores = np.squeeze(left_detection_keypoint_scores)
        right_detection_boxes = np.squeeze(right_detection_boxes)
        right_detection_scores= np.squeeze(right_detection_scores)
        right_detection_classes = np.squeeze(right_detection_classes)
        right_detection_keypoints = np.squeeze(right_detection_keypoints)
        right_detection_keypoint_scores = np.squeeze(right_detection_keypoint_scores)
        height,width = left_image.shape[0:2]
        composite = np.zeros_like(left_image)
        #print(left_detection_keypoints.shape,left_detection_keypoint_scores.shape)
        for (image,
             detection_boxes,
             detection_scores,
             detection_classes,
             detection_keypoints,
             detection_keypoint_scores) in [(left_image,
                                             left_detection_boxes,
                                             left_detection_scores,
                                             left_detection_classes,
                                             left_detection_keypoints,
                                             left_detection_keypoint_scores),
                                            (right_image,
                                             right_detection_boxes,
                                             right_detection_scores,
                                             right_detection_classes,
                                             right_detection_keypoints,
                                             right_detection_keypoint_scores)]:
            disp = image.copy()
            for box,score,claz,kps,confs in zip(detection_boxes,
                                                detection_scores,
                                                detection_classes,
                                                detection_keypoints,
                                                detection_keypoint_scores):
                if score < 0.3: continue
                if claz != 1: continue

                import human_foo
                human_foo.draw_human(disp,kps,confs)
                
            
        
            d = dict(detection_boxes=detection_boxes,
                     detection_scores=detection_scores,
                     detection_classes=detection_classes)
            disp = annotate(disp,d,score_thresh=0.3)
            composite += (disp/2).astype(np.uint8)
        imshow('lr composite',composite)
    vproc.playback_h5file(h5file,processor)


def preprocess(svo_filenames):
    #stages = [SVOExtractImages,
    #          DetectionCoCo,
    #          GlobalSegmentation,
    #          ]
    stages = [
        #RefineDetections,
        StoreCameraConfig,
        ]
    pipeline(stages,svo_filenames)

if __name__=='__main__':

    #svos = ['HD720_SN27165053_16-29-04.svo']
    #        'hug-1.svo',
    #        'depressed-drink-1.svo']
    #preprocess(svos)
    
    #d = DetectionCoCo('coco_centernet')
    #d.process_svo('HD720_SN27165053_16-29-04.svo')
    #extract_zed_images(['HD720_SN27165053_16-29-04.svo'])
    #s = GlobalSegmentation()
    #s.process_svo('HD720_SN27165053_16-29-04.svo')
    #playback_test6()
    #playback_test6point5()
    playback_test7()

##########################################################################
# frustratingly, after developing my own openpose-derived human pose     #
# estimating python module (tf2_pose) I found that it's not as good as   #
# the keypoints that come baked into one of the object detectors I'm     #
# using. This file contains useful visualizers and postprocessing        #
# helpers for human skeletons. tf2_pose, also installed on this docker   #
# image, may contain some other useful stuff.                            #
##########################################################################

import cv2
from enum import Enum
import numpy as np
from coco import coco
from util import shape_assert, _np, unbatch, assert_same_len
import filtering
from zed_geometry import to_3D


class Part(Enum):
    Nose = 0 #
    LEye = 1
    REye = 2
    LEar = 3
    REar = 4
    LShoulder = 5
    RShoulder = 6
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16
    Background = 17
    

Pairs = [
    (1,0),
    (2,0),
    (3,1),
    (4,2),
    (5,3),
    (6,4),
    (7,5),(9,7),
    (8,6),(10,8),
    (16,14),(14,12),(12,6),
    (15,13),(13,11),(11,5),
]  

Colors = [[255, 0, 0], #nose
          [255, 0, 255], #leye
          [170, 0, 255], #reye
          [255, 0, 85], #lear
          [255, 0, 170], #rear
          [85, 255, 0],  #lshoulder
          [255, 170, 0], #rshoulder
          [0, 255, 0],  #lelbow
          [255, 255, 0], #relbow
          [0, 255, 85], #lwrist
          [170, 255, 0], #rwrist
          [0, 85, 255], #lhip
          [0, 255, 170], #rhip
          [0, 0, 255], #lknee
          [0, 255, 255], #rknee
          [85, 0, 255], #lankle          
          [0, 170, 255], #rankle
         ]

def draw_human(img,joint_vect,confidence_vect,thresh=0.4,copy=False):
    if copy:
        img = img.copy()
    if img.shape[0]==1:
        img = img[0,...]
    if joint_vect.shape[0]==1:
        joint_vect = joint_vect[0,...]
    if confidence_vect.shape[0]==1:
        confidence_vect = confidence_vect[0,...]
        
    image_h,image_w = img.shape[0:2]
    centers = {}
    for i in range(Part.Background.value):
        if confidence_vect[i]<thresh:continue
        y,x = joint_vect[i,:]
        center = (int(x * image_w + 0.5),
                  int(y * image_h + 0.5))
        centers[i] = center
        cv2.circle(img, center, 3, Colors[i], thickness=3,
                   lineType=8, shift=0)
    for pair in Pairs:
        if any((pair[i] not in centers) for i in range(2)):
            continue
        cv2.line(img, centers[pair[0]],
                 centers[pair[1]], Colors[pair[0]], 3)
    return img

def draw_tracked(img,track_dict,copy=False,draw_bb=False,**draw_human_kwargs):
    if copy:
        img = img.copy()
    height,width = img.shape[0:2]
    import detector_foo
    color = detector_foo.coco_colors.get(1)
    for pid in track_dict:
        #img = np.zeros_like(img)
        (skel,scores,bbox,bbox_score) = track_dict[pid]
        draw_human(img,skel,scores,copy=False,**draw_human_kwargs)
        ymin, xmin, ymax, xmax = bbox
        ymin=int(ymin*height)
        ymax=int(ymax*height)
        xmin=int(xmin*width)
        xmax=int(xmax*width)
        if draw_bb:
            cv2.line(img,(xmin,ymin),(xmax,ymin),color,2)
            cv2.line(img,(xmax,ymin),(xmax,ymax),color,2)
            cv2.line(img,(xmax,ymax),(xmin,ymax),color,2)
            cv2.line(img,(xmin,ymax),(xmin,ymin),color,2)
        name = "person id:%d"%pid
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,name,(xmin,ymin), font, 0.5,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(img,name,(xmin,ymin), font, 0.5,color,1,cv2.LINE_AA)
        #cv2.imshow("debug",img)
        #cv2.waitKey()

def skel_dist(point_thresh,
              img_h,
              img_w,
              skelA, skelA_scores,
              skelB, skelB_scores,
              score_thresh = 0.01,
              min_valid_joints = 5):
    """
    given 2 skeletons, conf scores:
    out of all joint positions that have conf greater than score_thresh on both
    skeletons (--> max_j). if max_j is less than min_valid_joints, return infinity.
    
    find the total number of joints with distance less than point_thresh, divide
    that number by max_j (--> match_ratio). return 1 - match_ratio. 
    """
    skelA,skelA_scores,skelB,skelB_scores = \
                (np.squeeze(thing) for thing in (skelA,skelA_scores,
                                                 skelB,skelB_scores))
    #print('skel dist')
    #print('skel shape',skelA.shape,skelB.shape)
    #print('skel a',skelA,'skel b',skelB)
    skelA = skelA * np.array([img_h,img_w],dtype=np.float32)
    skelB = skelB * np.array([img_h,img_w],dtype=np.float32)
    JA = []
    JB = []
    
    for (ja,sa,jb,sb) in zip(skelA,skelA_scores,skelB,skelB_scores):
    #    print("sa",sa,"sb",sb)
        if sa > score_thresh and sb > score_thresh:
            JA.append(ja)
            JB.append(jb)
    #print("len(JA)",len(JA))
    if len(JA) < min_valid_joints:
        return 1.0
    close_pts = 0
    for (ja,jb) in zip(JA,JB):
        dist = np.sqrt(np.sum(np.square(ja-jb)))
    #    print("dist",dist)
        if dist < point_thresh:
            close_pts+=1
    #print("close points",close_pts)
    match_ratio = close_pts/float(len(JA))
    #print("match_ratio",match_ratio)
    return 1.0-match_ratio

def hit_test(box,point):
    shape_assert(box,[4])
    shape_assert(point,[2])
    ymin,xmin,ymax,xmax = box
    y,x = point
    return (ymin <= y <= ymax) and (xmin <= x <= xmax)

def median_skel_dist(skelA,skelB):
    """
    when noise is less of an issue, find the dist between two vects of skel
    joint positions, as the median distance between corresponding points
    """
    #epsilon = 1e-6
    euc = np.sqrt(np.sum(np.square(skelA-skelB),axis=-1))
    #return np.median(euc[euc>epsilon])
    return np.median(euc)

def is_duplicate_skeleton(skelA,skelA_scores,skelB,skelB_scores,point_dist_thresh,
                          total_dist_thresh=0.5):
    dist = skel_dist(point_dist_thresh,
                     720,#it shouldn't really matter that these dims are hard coded;
                     1280,#they're really just to give scale to the dist, in pix.
                     skelA,skelA_scores,
                     skelB,skelB_scores)
    return dist < total_dist_thresh

def skeleton_value(scores, cutoff=0.01):
    # := median score above cutoff
    shape_assert(scores,[17])
    if np.any(scores>cutoff):
        return np.median(scores[scores>cutoff])
    else:
        return 0

def remove_duplicate_skeletons(classes,
                               detection_scores,
                               skeletons,
                               skeleton_scores,
                               dist_thresh=20,
                               detect_score_thresh=0.1):
    shape_assert(classes,[1,None])
    shape_assert(detection_scores,[1,None])
    shape_assert(skeletons,[1,None,17,2])
    shape_assert(skeleton_scores,[1,None,17])
    (classes,detection_scores,skeletons,
     skeleton_scores) = (unbatch(arg).copy() for arg in (classes,
                                                         detection_scores,
                                                         skeletons,
                                                         skeleton_scores))
    assert_same_len(classes,detection_scores,skeletons,skeleton_scores)
    N = classes.shape[0]
    for i in range(N):
        if classes[i]!=coco.index('person'):continue
        if detection_scores[i] < detect_score_thresh:continue
        #it's a person with non-negligible score
        for j in range(i+1,N):
            if classes[j]!=coco.index('person'):continue
            if detection_scores[j] < detect_score_thresh:continue
            #it's also a person with non-negligible score
            if is_duplicate_skeleton(skeletons[i],skeleton_scores[i],
                                     skeletons[j],skeleton_scores[j],dist_thresh):
                if skeleton_value(skeleton_scores[i]) < \
                   skeleton_value(skeleton_scores[j]):
                    #wipe out skeleton i
                    skeleton_scores[i,:]=0
                else:
                    #wipe out skeleton j
                    skeleton_scores[j,:]=0
    skeleton_scores = skeleton_scores[np.newaxis,...]
    return skeleton_scores

def remove_low_val_skeletons(classes,skeleton_scores,cutoff = 0.2):
    shape_assert(classes,[1,None])
    shape_assert(skeleton_scores,[1,None,17])
    (classes,skeleton_scores) = (unbatch(arg).copy() for arg in (classes,
                                                                 skeleton_scores))
    assert_same_len(classes,skeleton_scores)
    N = classes.shape[0]
    for i in range(N):
        if classes[i]!=coco.index('person'):continue
        if skeleton_value(skeleton_scores[i],cutoff=0) < cutoff:
            #more than half of the joints in this skel have score less than cutoff
            skeleton_scores[i,...]=0
    return skeleton_scores[np.newaxis,...]

def box_fits_skel(box,skel):
    ymin,xmin,ymax,xmax = box
    skely = skel[:,0]
    skelx = skel[:,1]
    res =  np.all((skely>=ymin) &
                  (skely<=ymax) &
                  (skelx>=xmin) &
                  (skelx<=xmax))
    return res

def get_box_from_skel(skel,scores,t):
    skely = skel[:,0][scores>t]
    skelx = skel[:,1][scores>t]
    ymin = np.min(skely)
    ymax = np.max(skely)
    xmin = np.min(skelx)
    xmax = np.max(skelx)
    height = ymax - ymin
    width = xmax - xmin
    h_offset = 0.05*height
    w_offset = 0.025*width
    return np.array([ymin-h_offset,xmin-w_offset,
                     ymax+h_offset,xmax+h_offset])

def get_best_box_for_skel(skel,skel_score,boxes,scores):
    """
    get highest scoring box that fits skel, and return it and the score
    """
    best_score = -1
    best_box = None
    for (box,score) in zip(boxes,scores):
        if box_fits_skel(box,skel):
            #print("box fits skel")
            #print("score",score)
            if score > best_score:
                best_score=score
                best_box = box
    if best_box is None:
        best_box = get_box_from_skel(skel,skel_score,t=0.05)
        best_score = skeleton_value(skel_score,cutoff=0)
    return best_box,best_score

def remove_skel_joints_outside_box(skels,skel_scores,boxes):
    """
    this doesn't do much bc the boxes depend heavily on 
    skeleton joints.
    """
    shape_assert(skels,[1,None,17,2])
    shape_assert(skel_scores,[1,None,17])
    shape_assert(boxes,[1,None,4])
    skels,skel_scores,boxes = [
        unbatch(arg).copy() for arg in (skels,skel_scores,boxes)]
    assert_same_len(skels,skel_scores,boxes)
    for i in range(len(skels)):
        for j in range(len(skels[i])):
            point = skels[i][j]
            if not hit_test(boxes[i],point):
                skel_scores[i][j]=0
    return skel_scores[np.newaxis,...]

def reset_person_detections(classes,
                            detection_boxes,
                            detection_scores,
                            skeletons,
                            skeleton_scores,
                            cutoff=0.1):
    """
    set detection score to zero if skeleton is worthless

    XXX there's a problem with a previous processing step that sometimes 
    separates skeletons from their bounding boxes. need to search bounding boxes
    for right one for skelton, as a result.
    """
    shape_assert(classes,[1,None])
    shape_assert(detection_boxes,[1,None,4])
    shape_assert(detection_scores,[1,None])
    shape_assert(skeletons,[1,None,17,2])
    shape_assert(skeleton_scores,[1,None,17])
    (classes,
     detection_boxes,
     detection_scores,
     skeletons,
     skeleton_scores) = (unbatch(arg).copy() for arg in (classes,
                                                         detection_boxes,
                                                         detection_scores,
                                                         skeletons,
                                                         skeleton_scores))
    assert_same_len(classes,detection_scores,skeleton_scores)
    N = classes.shape[0]
    for i in range(N):
        if classes[i]!=coco.index('person'):continue
        if skeleton_value(skeleton_scores[i],cutoff=0) < cutoff:
            detection_scores[i] = 0
        else:
            box,score = get_best_box_for_skel(skeletons[i],
                                              skeleton_scores[i],
                                              detection_boxes,
                                              detection_scores)
            detection_boxes[i,:]=box
            detection_scores[i]=score
    return detection_boxes[np.newaxis,...],detection_scores[np.newaxis,...]

class SkeletonTracker(object):
    #SKEL_DIST_T = 0.5 #frac of valid points mismatching to still consider match
    SKEL_DIST_T = 0.15#median dist as fraction of image dim, generous
    FRAMES_TO_FORGET = 60 #after which we forget a person id, even if it might match
    IMPOSSIBLE_MOVEMENT_THRESH = 0.05 #no joint allowed to move this far between frame
    def __init__(self):
        self.tracks = {}#{id:{frame_num: (skel,scores,bbox,bbox_score)}}
        self.last_skels = {}#{id:(skel,scores,last_frame_seen)}
        self.sks = {}#{id: skel 2d Kalman filter apparatus}
        self.frame_counter=0
        self.person_counter=0

    def update(self,
               classes,
               detection_boxes,
               detection_scores,
               skeletons,
               skeleton_scores,
               cutoff=0.1,
               point_dist_thresh=30):
        shape_assert(classes,[1,None])
        shape_assert(detection_boxes,[1,None,4])
        shape_assert(detection_scores,[1,None])
        shape_assert(skeletons,[1,None,17,2])
        shape_assert(skeleton_scores,[1,None,17])
        (classes,detection_boxes,detection_scores,skeletons,
         skeleton_scores) = (unbatch(arg).copy() for arg in (classes,
                                                             detection_boxes,
                                                             detection_scores,
                                                             skeletons,
                                                             skeleton_scores))
        assert_same_len(classes,detection_boxes,
                        detection_scores,skeletons,skeleton_scores)
        N = classes.shape[0]
        for i in range(N):
            if classes[i]!=coco.index('person'):continue
            if skeleton_value(skeleton_scores[i],cutoff=0) < cutoff:continue
            skeleton,scores = skeletons[i],skeleton_scores[i]
            mindist = float('inf')
            closest_pid = None
            #print("len(last_skels)",len(self.last_skels))
            for (pid,(lskel,lscores,lframe)) in self.last_skels.items():
                if self.frame_counter-lframe < self.FRAMES_TO_FORGET:
                    #dist = skel_dist(
                    #    point_dist_thresh,
                    #    720,1280,
                    #    #^look the other way, these don't matter quite so much
                    #    skeleton,scores,
                    #    lskel,lscores)
                    dist = median_skel_dist(skeleton,lskel)
                    #print("pid",pid,"dist",dist)
                    if dist < mindist:
                        mindist = dist
                        closest_pid = pid
            if mindist < self.SKEL_DIST_T:

                if len(self.tracks[closest_pid]) >= 10: #give kalman filter time to
                    #catch up!
                    #then, reject skel joints that move too much btw frames.
                    skeleton,scores = self.prune_impossible_jumps(
                        skeleton,scores,
                        self.last_skels[closest_pid][0],#last seen skel
                        self.frame_counter - self.last_skels[closest_pid][2]
                        #^--frame diff
                    )

                sk = self.sks[closest_pid]
                skeleton,scores = sk(skeleton,scores)#apply kalman filtering
                #independently to skel joints, receive updated joints, confidence
                
                self.last_skels[closest_pid] = (skeleton,scores,self.frame_counter)
                self.tracks[closest_pid][self.frame_counter] = (
                    skeleton,
                    scores,
                    detection_boxes[i],
                    detection_scores[i])
            else:
                #new person seen
                #print ("new person",self.person_counter)
                self.sks[self.person_counter] = filtering.Pointwise2DSkelKF()
                skeleton,scores = self.sks[self.person_counter](skeleton,scores)
                
                self.last_skels[self.person_counter] = (skeleton,
                                                        scores,
                                                        self.frame_counter)
                self.tracks[self.person_counter] = {}
                self.tracks[self.person_counter][self.frame_counter] = (
                    skeleton,
                    scores,
                    detection_boxes[i],
                    detection_scores[i])
                self.person_counter += 1
        self.frame_counter +=1

    def prune_impossible_jumps(self,
                               new_skel,
                               new_scores,
                               old_skel,
                               frame_diff):
        for i in range(len(new_skel)):
            p2 = new_skel[i]
            p1 = old_skel[i]
            dist = np.sqrt(np.sum(np.square(p2-p1)))
            #print(i,":",dist)
            if dist > self.IMPOSSIBLE_MOVEMENT_THRESH*frame_diff:
                #new_skel[i,:]=p1
                new_scores[i] =0# *= 0.5**frame_diff
        return new_skel,new_scores

    def forget(self,person_id):
        """
        force stop tracking of person_id. a new id will be generated
        """
        del self.last_skels[person_id]
        del self.sks[person_id]

    def get_tracked(self):
        out = {}
        for (pid,td) in self.tracks.items():
            if self.frame_counter-1 in td:
                if len(td) > 15: #require half second of tracking, cumulative
                    #otherwise we output a lot of glitch skeletons.
                    out[pid] = td[self.frame_counter-1]
        return out
            
def filter_human(classes,
                 detection_boxes,
                 detection_scores,
                 skeletons,
                 skeleton_scores):
    shape_assert(classes,[1,None])
    shape_assert(detection_boxes,[1,None,4])
    shape_assert(detection_scores,[1,None])
    shape_assert(skeletons,[1,None,17,2])
    shape_assert(skeleton_scores,[1,None,17])
    (classes,detection_boxes,detection_scores,skeletons,
     skeleton_scores) = (unbatch(arg).copy() for arg in (classes,
                                                         detection_boxes,
                                                         detection_scores,
                                                         skeletons,
                                                         skeleton_scores))
    assert_same_len(classes,detection_boxes,
                    detection_scores,skeletons,skeleton_scores)
    mask = classes==coco.index('person')
    (classes,detection_boxes,detection_scores,skeletons,
     skeleton_scores) = (arg[mask] for arg in (classes,
                                               detection_boxes,
                                               detection_scores,
                                               skeletons,
                                               skeleton_scores))
    (classes,detection_boxes,detection_scores,skeletons,
     skeleton_scores) = (arg[np.newaxis,...] for arg in (classes,
                                                         detection_boxes,
                                                         detection_scores,
                                                         skeletons,
                                                         skeleton_scores))
    return (classes,detection_boxes,detection_scores,skeletons,
            skeleton_scores)

def align_lr_humans(left_args,right_args):
    for args in (left_args,right_args):
        assert np.all(args[0]==coco.index('person'))

    
    lboxes,lscores,lskels,lskel_scores = (unbatch(arg) for arg in left_args[1:])
    rboxes,rscores,rskels,rskel_scores = (unbatch(arg) for arg in right_args[1:])

    shape_assert(lboxes,[None,4])
    shape_assert(lscores,[None])
    shape_assert(lskels,[None,17,2])
    shape_assert(lskel_scores,[None,17])
    shape_assert(rboxes,[None,4])
    shape_assert(rscores,[None])
    shape_assert(rskels,[None,17,2])
    shape_assert(rskel_scores,[None,17])

    assert_same_len(lboxes,lscores,lskels,lskel_scores)
    assert_same_len(rboxes,rscores,rskels,rskel_scores)

    lmatches = {}#{ridx:(lidx,score)}

    epsilon = 1e-6 #zero testing skel score
    for i in range(len(lboxes)):
        lbox,lscore,lskel,lskel_score = (arg[i] for arg in (lboxes,lscores,
                                                            lskels,lskel_scores))
        if skeleton_value(lskel_score) < epsilon:
            continue
        min_dist = float('inf')
        for j in range(len(rboxes)):
            rbox,rscore,rskel,rskel_score = (arg[j] for arg in (rboxes,rscores,
                                                                rskels,rskel_scores))
            if skeleton_value(rskel_score) < epsilon:
                continue
            dist = median_skel_dist(lskel,rskel)
            if  dist < min_dist:
                min_dist = dist
                if j in lmatches:
                    k,kdist = lmatches[j]
                    if dist < k:
                        lmatches[j] = (i,dist)
                else:
                    if dist < SkelTracker3D.MAX_LR_DIST:
                        lmatches[j] = (i,dist)
    if not lmatches:
        zilch = (np.zeros((0,1),dtype = left_args[0].dtype),
                 np.zeros((0,4),dtype = lboxes.dtype),
                 np.zeros((0,17,2),dtype = lskels.dtype),
                 np.zeros((0,17),dtype = lskel_scores.dtype))
        return zilch,zilch
    r_idxs,lboop = zip(*lmatches.items())
    l_idxs,_ = zip(*lboop)
    
    lboxes,lscores,lskels,lskel_scores = (arg[list(l_idxs)] for arg in
                                          (lboxes,lscores,lskels,lskel_scores))
    rboxes,rscores,rskels,rskel_scores = (arg[list(r_idxs)] for arg in
                                          (rboxes,rscores,rskels,rskel_scores))
    lclasses = left_args[0][0,:len(lboxes)]
    rclasses = right_args[0][0,:len(rboxes)]

    #rebatch
    lclasses,lboxes,lscores,lskels,lskel_scores = (
        arg[np.newaxis,...] for arg in (lclasses,lboxes,lscores,lskels,lskel_scores))
    rclasses,rboxes,rscores,rskels,rskel_scores = (
        arg[np.newaxis,...] for arg in (rclasses,rboxes,rscores,rskels,rskel_scores))

    return (lclasses,lboxes,lscores,lskels,lskel_scores),\
        (rclasses,rboxes,rscores,rskels,rskel_scores)
                                    

def vertical_filter(left_args,right_args):
    """
    stereo disparity should be in x only for perfectly calibrated camera,
    which we can assume this is. drop confidence of misaligned joints, increase
    confidence of well-aligned joints, and average the y coordinate of all well
    aligned joints to increase accuracy.
    """
    for args in (left_args,right_args):
        assert np.all(args[0]==coco.index('person'))

    
    lboxes,lscores,lskels,lskel_scores = (unbatch(arg) for arg in left_args[1:])
    rboxes,rscores,rskels,rskel_scores = (unbatch(arg) for arg in right_args[1:])

    shape_assert(lboxes,[None,4])
    shape_assert(lscores,[None])
    shape_assert(lskels,[None,17,2])
    shape_assert(lskel_scores,[None,17])
    shape_assert(rboxes,[None,4])
    shape_assert(rscores,[None])
    shape_assert(rskels,[None,17,2])
    shape_assert(rskel_scores,[None,17])

    assert_same_len(lboxes,lscores,lskels,lskel_scores,
                    rboxes,rscores,rskels,rskel_scores)
    N = len(lskels)
    for i in range(N):
        l_y = lskels[i,:,0]
        r_y = rskels[i,:,0]
        y_abs_diff = np.abs(l_y-r_y)
        lskel_scores[i,y_abs_diff>SkelTracker3D.VERTICAL_MISALIGN_THRESH] = 0
        rskel_scores[i,y_abs_diff>SkelTracker3D.VERTICAL_MISALIGN_THRESH] = 0
        l_y[...] = r_y[...] = (l_y+r_y)/2
    #no return; modify args in place
    

def prefilter(detection_classes,
              detection_boxes,
              detection_scores,
              detection_keypoints,
              detection_keypoint_scores):
    detection_keypoint_scores = remove_duplicate_skeletons(
        detection_classes,
        detection_scores,
        detection_keypoints,
        detection_keypoint_scores)
    detection_keypoint_scores = remove_low_val_skeletons(
        detection_classes,
        detection_keypoint_scores)
    detection_boxes,detection_scores = reset_person_detections(
        detection_classes,
        detection_boxes,
        detection_scores,
        detection_keypoints,
        detection_keypoint_scores)
    return (detection_classes,
            detection_boxes,
            detection_scores,
            detection_keypoints,
            detection_keypoint_scores)

class SkelTracker3D(object):
    MAX_LR_DIST = 0.05 #max distance (fractional) for stereo skels
    VERTICAL_MISALIGN_THRESH = 0.02 #max vertical mismatch of l,r joints
    def __init__(self):
        self.left_tracker = SkeletonTracker()
        self.right_tracker = SkeletonTracker()
        self.frame_counter=0
        self.person_counter=0
        self.tracks = {}#{person_id: {frame: 3d_skel, left_2d_id, right_2d_id}}
        

    def update(self,left_args,right_args,camera_info):

        left_args = prefilter(*left_args)
        right_args= prefilter(*right_args)

        #remove non-human bounding boxes so we don't waste time iterating
        left_args,right_args = (filter_human(*args)
                                for args in (left_args,right_args))


        left_args,right_args = align_lr_humans(left_args,right_args)
        
        if len(left_args[0])==0:
            return

        vertical_filter(left_args,right_args)#in-place modify

        #here, could convert to 3D and filter out skeletons with insane bone lengths
        #leave out for simplicity, unless seems like it'd help...
        
        ##sanity check:
        #lskels = left_args[3][0]
        #lskscores=left_args[4][0]
        #rskels = right_args[3][0]
        #rskscores=right_args[4][0]
        #assert_same_len(lskels,lskscores,rskels,rskscores)
        #for i in range(len(lskels)):
        #    z = np.zeros((720,1280,3),dtype=np.uint8)
        #    print(lskscores[i],rskscores[i])
        #    draw_human(z,lskels[i],lskscores[i])
        #    draw_human(z,rskels[i],rskscores[i])
        #    cv2.imshow("debug",z)
        #    cv2.waitKey()
        
        self.left_tracker.update(*left_args)
        self.right_tracker.update(*right_args)

        #debug / demo
        ll = self.left_tracker.last_skels
        rl = self.right_tracker.last_skels

        if len(rl)==1 and len(ll) == 1:
            lskel,lscores,_ = ll[0]
            rskel,rscores,_ = rl[0]
            scores = (lscores+rscores)/2
            s3d = to_3D(lskel,rskel,camera_info)
            return s3d,scores
        else:
            return [],[]
            

        #sanity check:
        # z = np.zeros((720,1280,3),dtype=np.uint8)
        # lskels = left_args[3][0]
        # lskscores=left_args[4][0]
        # rskels = right_args[3][0]
        # rskscores=right_args[4][0]
        # for i in range(len(lskels)):
        #     draw_tracked(z,self.left_tracker.get_tracked(),thresh=0.2)
        #     draw_tracked(z,self.right_tracker.get_tracked(),thresh=0.2)
        #     cv2.imshow("debug",z)

        #assert self.left_tracker.frame_count == self.right_tracker.frame_count
        #self.frame_counter = self.left_tracker.frame_counter

        
                                                                      

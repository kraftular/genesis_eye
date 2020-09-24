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


def draw_human(img,joint_vect,confidence_vect,thresh=0.2,copy=False):
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

def draw_tracked(img,track_dict,copy=False,**draw_human_kwargs):
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

def get_box_from_skel(skel):
    skely = skel[:,0]
    skelx = skel[:,1]
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
        best_box = get_box_from_skel(skel)
        best_score = skeleton_value(skel_score,cutoff=0)
    return best_box,best_score

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
    SKEL_DIST_T = 0.5 #frac of valid points mismatching to still consider match
    FRAMES_TO_FORGET = 60 #after which we forget a person id, even if it might match
    def __init__(self):
        self.tracks = {}#{id:{frame_num: (skel,scores,bbox,bbox_score)}}
        self.last_skels = {}#{id:(skel,scores,last_frame_seen)}
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
                    dist = skel_dist(
                        point_dist_thresh,
                        720,1280,
                        #^look the other way, these don't matter quite so much
                        skeleton,scores,
                        lskel,lscores)
                    #print("pid",pid,"dist",dist)
                    if dist < mindist:
                        mindist = dist
                        closest_pid = pid
            if mindist < self.SKEL_DIST_T:
                
                self.last_skels[closest_pid] = (skeleton,scores,self.frame_counter)
                self.tracks[closest_pid][self.frame_counter] = (
                    skeleton,
                    scores,
                    detection_boxes[i],
                    detection_scores[i])
            else:
                #new person seen
                #print ("new person",self.person_counter)
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

    def forget(self,person_id):
        """
        force stop tracking of person_id. a new id will be generated
        """
        del self.last_skels[person_id]

    def get_tracked(self):
        out = {}
        for (pid,td) in self.tracks.items():
            if self.frame_counter-1 in td:
                out[pid] = td[self.frame_counter-1]
        return out
            

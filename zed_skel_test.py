import pyzed.sl as sl
import cv2
import numpy as np
from tf2_pose.estimator import TfPoseEstimator
from tf2_pose.networks import get_graph_path
import subprocess

import tensorflow_hub as hub


from detector_foo import Gate,BatchObjectDetector,annotate


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        #for gpu in gpus:
        #    tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], 
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
        
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)



# def annotate(img,drd):
#     boxes,classes,scores = [np.squeeze(drd["detection_"+i])
#                             for i in "boxes classes scores".split()]
#     height,width = img.shape[0:2]
#     #print(boxes.shape,classes.shape,scores.shape)
#     for box,claz,score in zip(boxes,classes,scores):
#         ymin, xmin, ymax, xmax = box
#         ymin=int(ymin*height)
#         ymax=int(ymax*height)
#         xmin=int(xmin*width)
#         xmax=int(xmax*width)
#         if score < 0.5: continue
#         if True:#claz==0:#person
#             cv2.line(img,(xmin,ymin),(xmax,ymin),(0,255,0))
#             cv2.line(img,(xmax,ymin),(xmax,ymax),(0,255,0))
#             cv2.line(img,(xmax,ymax),(xmin,ymax),(0,255,0))
#             cv2.line(img,(xmin,ymax),(xmin,ymin),(0,255,0))
#             name = coco[int(claz)]
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             cv2.putText(img,name,(xmin,ymin), font, 0.5,(255,255,255),2,cv2.LINE_AA)


# def annotate_oi(img,drd):
#     #print(drd)
#     boxes,classes,scores = [np.squeeze(drd["detection_"+i])
#                             for i in "boxes class_entities scores".split()]
#     height,width = img.shape[0:2]
#     #print(boxes.shape,classes.shape,scores.shape)
#     for box,claz,score in zip(boxes,classes,scores):
#         ymin, xmin, ymax, xmax = box
#         ymin=int(ymin*height)
#         ymax=int(ymax*height)
#         xmin=int(xmin*width)
#         xmax=int(xmax*width)
#         if score < 0.3: continue
#         if True:#claz==0:#person
#             cv2.line(img,(xmin,ymin),(xmax,ymin),(0,255,0))
#             cv2.line(img,(xmax,ymin),(xmax,ymax),(0,255,0))
#             cv2.line(img,(xmax,ymax),(xmin,ymax),(0,255,0))
#             cv2.line(img,(xmin,ymax),(xmin,ymin),(0,255,0))
#             name = claz.decode('utf-8')
#             print(name)
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             cv2.putText(img,name,(xmin,ymin), font, 0.5,(255,255,255),2,cv2.LINE_AA)
        

def main():

    #warm up tensorflow??
    #h = HumanPoseEstimator()
    #h.get_humans(imgs=np.zeros((1,720,1280,3),dtype=np.uint8))

    est =  TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
    est.inference(np.zeros((1,720,1280,3),dtype=np.uint8), resize_to_default=True,
                                   upsample_size=4)

    print(subprocess.check_output("nvidia-smi").decode('utf-8'))

    print("loading model")
    #detector = hub.load("https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1")#slow
    #detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")#inaccurate
    #detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1")
    #detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_800x1333/1")#ok
    #detector = hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1")#slow
    #detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_800x1333/1")
    #detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d4/1")
    #detector = hub.load(
    #    "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")
    #detector = hub.load(
    #    "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1")
    #detector = detector.signatures['default']


    gate = Gate([BatchObjectDetector()])

    def detect(frame):
        out = gate.push(frame)
        if out:
            frame, (d,) = out
            return frame,d
        else:
            return None,None
    
    
    print("done")


    
    
    print("warming up detector")
    #detector(tf.constant(np.ones((1,720,1280,3),dtype=np.uint8)))#warm up
    #detector(tf.constant(np.ones((1,720,1280,3),dtype=np.float32)))#warm up
    print("done")

    print(subprocess.check_output("nvidia-smi").decode('utf-8'))
    
    #have to warm up tensorflow, else zed will take all its memory or something; tf
    #will fail to execute cuda 
    
    zed = sl.Camera()

    print("created zed")


    
    
    # Create a InitParameters object and set configuration parameters

    input_type = sl.InputType()
    input_type.set_from_svo_file("~/genesis_eye/videos/HD720_SN27165053_16-29-04.svo")
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    init_params.sdk_verbose = False
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA


    print(subprocess.check_output("nvidia-smi").decode('utf-8'))
    
    print("about to open zed")

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    print("opened zed")
        
    # # Set initialization parameters
    # detection_parameters = sl.ObjectDetectionParameters()
    # detection_parameters.enable_tracking = True

    # # Set runtime parameters
    # detection_parameters_rt = sl.ObjectDetectionRuntimeParameters()
    # detection_parameters_rt.detection_confidence_threshold = 25

    # if detection_parameters.enable_tracking :
    #     # Set positional tracking parameters
    #     positional_tracking_parameters = sl.PositionalTrackingParameters()
    #     positional_tracking_parameters.set_floor_as_origin = True

    #     # Enable positional tracking
    #     zed.enable_positional_tracking(positional_tracking_parameters)


    
    
    
    # Enable object detection with initialization parameters
    #zed_error = zed.enable_object_detection(detection_parameters)
    #if zed_error != sl.ERROR_CODE.SUCCESS :
    #    print("enable_object_detection", zed_error, "\nExit program.")
    #    zed.close()
    #    exit(-1)

    

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    # objects = sl.Objects()
    key = ''
    while key != 'q':
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(mat,sl.VIEW.LEFT)
            img = mat.get_data()[...,:3]
            frame = img
            #frame = tf.image.resize(frame,[360,640]).numpy().astype(np.uint8)
            frame=frame.astype(np.uint8)
            img=frame

            #imgs = tf.constant(np.array([img]).astype(np.uint8)[...,::-1])
            imgs = tf.constant(np.array([img]).astype(np.float32)[...,::-1]/255)


            #print(imgs.shape)

            #maps = h.process_raw(imgs)
            #humanss = h.get_humans(imgs=imgs)

            #renders = h.draw_humans(imgs,humanss)

            #humans = est.inference(img, resize_to_default=True,
            #                       upsample_size=4)

            #detector_output = detector(imgs)

            out_frame,d = detect(img.astype(np.float32))

            
            
            #print(detector_output)

            #TfPoseEstimator.draw_humans(img, humans, imgcopy=False)

            # zed.retrieve_objects(objects, detection_parameters_rt)

            #zed object detection is shit
            # for obj in objects.object_list:
            #     print(obj.label)
            #     if obj.confidence < 0.5 :
            #         continue
            #     object_2Dbbox = obj.bounding_box_2d
            #     a,b,c,d = map(lambda l:tuple(int(i) for i in l),object_2Dbbox)
            #     cv2.line(img,a,b,(0,0,255))
            #     cv2.line(img,b,c,(0,0,255))
            #     cv2.line(img,c,d,(0,0,255))
            #     cv2.line(img,d,a,(0,0,255))
            #     mask = obj.mask.get_data()
            #     window = img[a[1]:d[1],a[0]:b[0],:]
            #     window[mask>0] = (0.5*window[mask>0] + 127).astype(np.uint8)

            if d is not None:
                out_frame = out_frame.astype(np.uint8)
                annotate(out_frame,d)
                cv2.imshow("img",out_frame)

            #annotate_oi(img,detector_output)
                
            #cv2.imshow("img",img)
        key = chr(cv2.waitKey(1)&0xff)


coco = """__bg
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
street sign
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
hat
backpack
umbrella
shoe
eye glasses
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
plate
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
mirror
dining table
window
desk
toilet
door
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
blender
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
hair brush""".split('\n')

coco_blacklist = [
    'tv',
    'bed',
    ]

if __name__=='__main__':
    main()

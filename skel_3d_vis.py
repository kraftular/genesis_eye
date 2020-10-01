#!/usr/bin/python

#ADK skeleton visualization program for Emerald json data
#this script works with python 2 and 3

#some helpful error messages for uncommon imported packages:
try:
    import pygame
    from pygame.locals import *
except ImportError:
    print("pygame is missing. try pip install pygame or apt-get install python-pygame")
    quit()

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except ImportError:
    print("python-opengl is missing. try apt-get install python-opengl")
    quit()

import json
import numpy as np
import sys
import inspect
import functools
from types import MethodType
import argparse
from datetime import datetime

from human_foo import Part,Pairs,Colors
import human_foo
from util import shape_assert, assert_same_len
import vproc



colors3f = np.array(Colors,dtype=np.float32)/255


def draw_skel(skeleton,confidence, thresh=0.4):
    if skeleton.shape[0]==1:
        skeleton = skeleton[0,...]
    if confidence.shape[0]==1:
        confidence = confidence[0,...]
    shape_assert(skeleton,[17,3])
    shape_assert(confidence,[17])
    centers = {}
    glColor3fv((0.0,1.0,0.0))
    for i in range(Part.Background.value):
        if confidence[i]<thresh:continue
        x,y,z = skeleton[i,:]
        #print(x,y,z)
        vect = np.array([x,y,z])
        centers[i] = vect
        glPushMatrix()
        glTranslatef(*vect) #move to where we want to put object
        glutSolidSphere(0.05,20,20) # make sphere (radius, res, res)
        glPopMatrix()
    glColor3fv((1.0,0.0,0.0))
    glBegin(GL_LINES)
    for pair in Pairs:
        if any((pair[i] not in centers) for i in range(2)):
            continue
        for vertex in (centers[i] for i in pair):
            glVertex3fv(vertex)
    glEnd()


def draw_grid(y0):
    """
    GL draw dancefloor in pygame's event loop
    """
    glColor3fv((0.5,0.5,0.5))
    glBegin(GL_LINES)
    for i in range(11):
        v1 = np.array([i-5,y0,0],dtype=np.float32)
        v2 = np.array([i-5,y0,10],dtype=np.float32)
        glVertex3fv(tuple(v1))
        glVertex3fv(tuple(v2))
        v3 = np.array([-5,y0,i],dtype=np.float32)
        v4 = np.array([5,y0,i],dtype=np.float32)
        glVertex3fv(tuple(v3))
        glVertex3fv(tuple(v4))
    glEnd()


def rotation_matrix(axis, theta_degrees):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta_degrees degrees.
    """
    theta = theta_degrees*np.pi/180.0
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def recorded(f):
    """
    deco for making a method in a subclass of Recorder 
    be recorded. this is used for saving a config file.
    """
    f.record=True
    return f

class Recorder(object):
    """
    subclasses of me have our @recorded methods recorded, that is,
    name, args, kwargs are noted so that they can be replayed.

    this is used in saving a configuration (camera angle, etc.)
    to a file.
    """
    #note that this class has nothing to do with recording videos,
    #which is handled by OpenCV below. 
    def __init__(self):
        self.journal = []
        self.last_save=None
        self.last_serialized=None
        methods = inspect.getmembers(self.__class__,predicate=inspect.ismethod)
        for (name,method) in methods:
            if hasattr(method,"record"):
                wrapped = self.wrap(method)
                setattr(self,name,wrapped)
    def wrap(self,method):
        @functools.wraps(method)
        def wrapped(soolf,*args,**kwargs):
            self.record(method.__name__,args,kwargs)
            return method(self,*args,**kwargs)
        wrapped = MethodType(wrapped,self,self.__class__)
        return wrapped
    def record(self,name,arglist,kwargd):
        self.journal.append((name,arglist,kwargd))
    def save_snapshot(self):
        """
        saves a snapshot of the current configuration
        that can be loaded (replayed) later
        """
        serialized = json.dumps(self.journal)
        if serialized==self.last_serialized:
            print("already saved current config in %s"%self.last_save)
            return
        else:
            self.last_serialized=serialized
        now = datetime.now()
        self.last_save=fname=now.strftime("snapshot-%Y-%m-%d-%H-%M-%S.config")
        with open(fname,'w') as snap:
            snap.write(serialized)
        print("saved %s"%fname)
    def load_snapshot(self,fname):
        with open(fname,'r') as snap:
            ser = snap.read()
            j = json.loads(ser)
            self.last_serialized=ser
            self.last_save=fname
            for entry in j:
                self.replay(entry)
                
        print("loaded configuration from %s"%fname)
    def replay(self,jentry):
        name,args,kwargs = jentry
        method = getattr(self,name)
        return method(*args,**kwargs)
        
        

class Console(Recorder):
    """
    implement interactions with user during playback. methods marked 
    @recorded are remembered as a sequence of actions that can be 
    played back. The configuration files saved by the program are
    simply serialized lists of such actions. 
    """
    grid=False
    paused=False
    lr_vect  = np.array([1,0,0],dtype=np.float32)#axis to translate left and right
    ud_vect  = np.array([0,1,0],dtype=np.float32)#axis to translate up and down
    io_vect  = np.array([0,0,1],dtype=np.float32)#axis to translate in and out
    grid_pos = -1.15#position along Y axis to draw dancefloor
    def print_commands(self):
        c = """
        keyboard commands:
              up/down arrow:           pan up and down
              right/left arrow:        pan left and right
              shift-up/shift-down:     pitch
              shift-left/shift-right:  yaw
              ctrl-up/ctrl-down:       zoom
              ctrl-left/ctrl-right:    roll
              spacebar:                pause/unpause
              g:                       dancefloor grid on/grid off
              ctrl-g:                  move grid up 5 centimeters
              shift-g:                 move grid down 5 centimeters
              s:                       save snapshot of current config into working dir
                                       [snapshots can be reloaded with --config option]
              q:                       quit
        """
        print(c)
    def update_vects(self,rotmat):
        self.lr_vect = np.dot(rotmat,self.lr_vect)
        self.ud_vect = np.dot(rotmat,self.ud_vect)
        self.io_vect = np.dot(rotmat,self.io_vect)
    def processEvent(self,event):
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        elif event.type==pygame.KEYDOWN:
            if event.mod & pygame.KMOD_SHIFT:
                if event.key == pygame.K_UP:
                    self.rotX(1)
                elif event.key == pygame.K_DOWN:
                    self.rotX(-1)
                elif event.key == pygame.K_LEFT:
                    self.rotY(1)
                elif event.key == pygame.K_RIGHT:
                    self.rotY(-1)
                elif event.key == pygame.K_g:
                    self.move_dancefloor(0.05)
            elif event.mod & pygame.KMOD_CTRL:
                if event.key == pygame.K_UP:
                    self.goIn(0.1)
                elif event.key == pygame.K_DOWN:
                    self.goOut(0.1)
                elif event.key == pygame.K_LEFT:
                    self.rotZ(1)
                elif event.key == pygame.K_RIGHT:
                    self.rotZ(-1)
                elif event.key == pygame.K_g:
                    self.move_dancefloor(-0.05)
            else:
                if event.key == pygame.K_UP:
                    self.goUp(0.1)
                elif event.key == pygame.K_DOWN:
                    self.goDown(0.1)
                elif event.key == pygame.K_LEFT:
                    self.goLeft(0.1)
                elif event.key == pygame.K_RIGHT:
                    self.goRight(0.1)
                elif event.key == pygame.K_g:
                    self.toggleGrid()
                elif event.key == pygame.K_SPACE:
                    self.togglePaused()
                elif event.key == pygame.K_s:
                    self.save_snapshot()
                elif event.key == pygame.K_q:
                    print("quit")
                    quit()
    def draw(self,skels, confidences):
        if skels is not None:
            shape_assert(skels,[None,17,3])
            shape_assert(confidences,[None,17])
            assert_same_len(skels,confidences)
        else:
            skels,confidences = [],[]
        if self.grid:
            draw_grid(self.grid_pos)
        for skel,c in zip(skels, confidences):
            draw_skel(skel,c)

    @recorded
    def move_dancefloor(self,how_much):
        self.grid_pos += how_much
    @recorded
    def goUp(self,meters):
        glTranslatef(*tuple(self.ud_vect*-meters))
    @recorded
    def goDown(self,meters):
        glTranslatef(*tuple(self.ud_vect*meters))
    @recorded
    def goLeft(self,meters):
        glTranslatef(*tuple(self.lr_vect*meters))
    @recorded
    def goRight(self,meters):
        glTranslatef(*tuple(self.lr_vect*-meters))
    @recorded
    def goIn(self,meters):
        glTranslatef(*tuple(self.io_vect*meters))
    @recorded
    def goOut(self,meters):
        glTranslatef(*tuple(self.io_vect*-meters))
    @recorded
    def toggleGrid(self):
        self.grid=not self.grid
    def togglePaused(self):
        self.paused=not self.paused
    @recorded
    def rotZ(self,ang):
        rotmat = rotation_matrix(self.io_vect,-ang)
        self.update_vects(rotmat)
        glRotatef(ang, *tuple(self.io_vect))
    @recorded
    def rotY(self,ang):
        rotmat = rotation_matrix(self.ud_vect,-ang)
        self.update_vects(rotmat)
        glRotatef(ang, *tuple(self.ud_vect))
    @recorded
    def rotX(self,ang):
        rotmat = rotation_matrix(self.lr_vect,-ang)
        self.update_vects(rotmat)
        glRotatef(ang, *tuple(self.lr_vect))
        
class Done(Exception):
    pass
        
class SkelFileWrapper(object):
    def __init__(self,f):
        self.f=f
        self.mode=None
        self.next_idx=0
        self._try_guess_mode()
    def _try_guess_mode(self):
        if self.f is sys.stdin:
            self.mode='stream'
    def next_frame(self):
        self._try_guess_mode()
        if self.mode is None:
            o = json.loads(self.f.readline())
            if type(o) is not list or len(o) < 1:
                raise ValueError("the json file or stream must yield list of list of dict, or list of dict.")
            if type(o[0]) is list:
                self.mode='all'
                self.all_frames = o
                return o[self.next_idx]
            elif type(o[0]) is dict:
                self.mode='stream'
                return o
            else:
                raise ValueError("bad input json.")
        elif self.mode=='all':
            self.next_idx+=1
            try:
                return self.all_frames[self.next_idx]
            except IndexError:
                raise Done("end of input file")
        elif self.mode=='stream':
            self.next_idx+=1
            try:
                return json.loads(self.f.readline())
            except ValueError:
                raise Done("either end of stream reached, or corrupt json data in stream.")
        else:
            raise ValueError("bad mode! %s"%mode)
    def seek(self,idx):
        if self.mode=='all':
            self.next_idx = idx%(len(self.all_frames))
        else:
            raise ValueError("seek is unsupported in %s mode"%self.mode)
   
        
def main():
    parser = argparse.ArgumentParser(description="ADK's skeleton-json visualization program")
    parser.add_argument('--size',dest='size',type=str,nargs='?',help="window size, default is 800x600",default="800x600")
    parser.add_argument('--fps',dest='fps',type=float,nargs='?',help="frames per second. default is 15",default=15)
    parser.add_argument('--video-file',dest='vfile',type=str,nargs='?',help="video file to save session to",default=None)
    parser.add_argument('--video-codec',dest='vcodec',type=str,nargs='?',help="video codec to use if saving video. default is MJPG",default='MJPG')
    parser.add_argument('--config',dest='config',type=str,nargs='?',help="file to recover configuration from prev. session",default=None)
    parser.add_argument('--svo',dest='svo',type=str,help="svo file name")

    args = parser.parse_args()
    
    pygame.init()

    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glClearColor(0.,0.,0.,1.)


    dshape = tuple(map(int,args.size.lower().split("x")))
   
    
    display = dshape
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    pygame.display.set_caption("skeleton viewer")

    pygame.key.set_repeat(50, 50)

    gluPerspective(45, (float(display[0])/float(display[1])), 0.1, 50.0)

    glTranslatef(0.0,0.0, -15)

    console = Console()
    console.print_commands()
    if args.config:
        console.load_snapshot(args.config)

    h5file = vproc.get_hdf5_for_svo(args.svo)
        
    if args.vfile:
        try:
            import cv2
        except ImportError:
            print("video recording requires opencv: pip install opencv-python")
            quit()
        fourcc = cv2.VideoWriter_fourcc(*args.vcodec)
        vwriter = cv2.VideoWriter(args.vfile, fourcc, args.fps,
			dshape, True)
        print("will write video frames to %s (codec: %s)"%(args.vfile,args.vcodec))

    wait_ms = int(1000/args.fps)
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
        loop_start = pygame.time.get_ticks()
        update_out = st.update(left_args,right_args,camera_info)
        if update_out is None:
            skels,confs = None,None
        else:
            skels,confs = update_out
            skels = skels[np.newaxis,...]
            confs = confs[np.newaxis,...]
        for event in pygame.event.get():
            console.processEvent(event)
            
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        console.draw(skels,confs)
        if args.vfile:
            raw_buf = pygame.image.tostring(pygame.display.get_surface(), 'RGB')
            img_array = np.frombuffer(raw_buf,dtype=np.uint8).reshape(
                (dshape[::-1]+(3,)))[:,:,::-1]
            vwriter.write(img_array)
        
        pygame.display.flip()
        loop_ms = int((pygame.time.get_ticks() - loop_start)/1000.0)
        pygame.time.wait(max(1,wait_ms-loop_ms))

    try:
        vproc.playback_h5file(h5file,processor,array_args=['camera_info'])
    finally:
        if args.vfile:
            vwriter.release()


if __name__=='__main__':
    main()

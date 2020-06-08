import os
import cv2
import numpy as np
from glob import glob
import argparse
import subprocess
import logging


CV_CAMERA_PRODUCT_NAME="HD Pro Webcam C920"#change this if you can't get an
#old-fashioned Carl Zeiss Tessar HD 1080p Logitech camera. these were cheap
#and plentiful before the pandemic, and may be again someday... all my code
#is calibrated to use them so they're the best option. but if you can't find
#one you can find the product name of your camera with:
#
#$ /bin/udevadm info --name=/dev/videoFOO |grep ID_V4L_PRODUCT

POWER_LINE_FREQUENCY = "60 Hz"#used by camera firmware, not us. other valid values are "50 Hz" or
                              #"Disabled".

def get_cv_webcams():
    video_devs = glob("/dev/video*")
    webcams = []
    for dev in video_devs:
        udev_info = subprocess.run(("/bin/udevadm info --name=%s"%dev).split(),
                                   stdout=subprocess.PIPE).stdout.decode('utf-8')
        udev_e = dict(line.strip()[3:].split('=') for line in udev_info.split('\n')
                      if line.startswith("E: "))
        if udev_e["ID_V4L_PRODUCT"]==CV_CAMERA_PRODUCT_NAME:
            logging.info("found cv webcam at %s"%dev)
            meta = dict(udev_e)
            meta['device_path']=dev
            webcams.append(meta)
            #nb unique id for cam is ID_SERIAL
    return webcams

class CamCtl(object):
    def __init__(self,webcam):
        if type(webcam) is dict:
            self.device = udev_dict['device_path']
        elif type(webcam) is str:
            self.device = webcam
        elif type(webcam) is int:
            self.device = "/dev/video%d"%webcam
        else:
            raise ValueError(webcam)
        self.controls = {}
        ctrl_lines = subprocess.run(("v4l2-ctl --device=%s --list-ctrls-menus"%self.device).split(),
                                    stdout=subprocess.PIPE).stdout.decode('utf-8')\
                               .strip().split('\n')
        self.controls = {}
        last_menu=None
        for line in ctrl_lines:
            if line.startswith('\t'):
                number,item = map(lambda x:x.strip(),line.strip().split(":"))
                last_menu['item']=number
            else:
                ctrl = {}
                tokens = line.strip().split()
                ctrl['type']=tokens[2][1:-1]
                ctrl['address'] = tokens[1]
                ctrl['range'] = {}
                for t in tokens[4:]:
                    key,val = t.split('=')
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                    ctrl['range'][key]=val
                if ctrl['type']=='menu':
                    ctrl['menu'] = last_menu = {}
                self.controls[tokens[0]]=ctrl
        self.focus_at_infinity()
    def set(self,key,val):
        ctrl = self.controls[key]
        if ctrl['type']=='menu':
            val = ctrl['menu'][val]
        val = str(val)
        cmd_out = subprocess.run(("v4l2-ctl --device=%s --set-ctrl=%s=%s"%(self.device,
                                                                           key,
                                                                           val)).split(),
                                 stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
        if cmd_out:
            raise ValueError(cmd_out)
        return self.get(key)
    def get(self,key):
        cmd_out = subprocess.run(("v4l2-ctl --device=%s --get-ctrl=%s"%(self.device,
                                                                           key)).split(),
                                 stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
        if not cmd_out.startswith(key):
            raise ValueError(cmd_out)
        v = cmd_out.split(": ")[1]
        self.controls[key]['range']['value']=int(v)
        return v
    def focus_at_infinity(self):
        self.set('focus_auto',0)
        self.set('focus_absolute',0)

if __name__=='__main__':
    logging.getLogger().setLevel(logging.INFO)
    print(get_cv_webcams())
    c = CamCtl(1)
    print(c.get("focus_absolute"))

import pyzed.sl as sl
import cv2
import numpy as np



def extract_params(svo):
    zed = sl.Camera()
    input_type = sl.InputType()
    input_type.set_from_svo_file(svo)
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False,
                                    coordinate_units=sl.UNIT.METER)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise ValueError("zed")

    ci = zed.get_camera_information()
    cal_param = ci.calibration_parameters
    R = cal_param.R.copy()
    T = cal_param.T.copy()

    #sanity check
    epsilon = 1e-12
    assert np.all(R<epsilon),"non-zero rotation vector! is camera calibrated?"
    assert np.all(T[1:] < epsilon),"non-zero translation in y or z!"

    baseline = T[0]

    left_params = cal_param.left_cam
    right_params = cal_param.right_cam

    assert left_params.fx == left_params.fy, "left focus not calibrated!"
    assert right_params.fx == right_params.fy, "right focus not calibrated!"

    assert left_params.fx == right_params.fx, "stereo focus not calibrated!"

    flen = left_params.fx

    left_optical_center = left_params.cy, left_params.cx
    right_optical_center = right_params.cy, right_params.cx

    camera_info = np.array([baseline,
                            flen,
                            left_optical_center[0],
                            left_optical_center[1],
                            right_optical_center[0],
                            right_optical_center[1]
                            ])
    return camera_info

def to_3D(left_points,right_points,camera_info,img_dims=(720,1280)):

    img_h,img_w = img_dims

    (baseline,
     flen,
     left_optical_center_y,
     left_optical_center_x,
     right_optical_center_y,
     right_optical_center_x) = camera_info
    
    
    left_x = left_points[...,1] *img_w - left_optical_center_x
    left_y = left_optical_center_y - left_points[...,0] *img_h
    right_x = right_points[...,1] * img_w - right_optical_center_x
    right_y = right_optical_center_y - right_points[...,0] * img_h

    dx = (left_x - right_x)

    avg_y = (left_y+right_y)/2

    print("baseline",baseline)
    print("dx",dx)

    z = baseline*flen / dx

    y = baseline*avg_y / dx

    x = baseline*(left_x + left_y) / (2*dx)

    return np.stack([x,y,z],axis=-1)
    


def test():
    zed = sl.Camera()
    input_type = sl.InputType()
    input_type.set_from_svo_file("~/genesis_eye/videos/HD720_SN27165053_16-29-04.svo")
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise ValueError("zed")

    ci = zed.get_camera_information()
    cal_param = ci.calibration_parameters
    print(cal_param.R)
    print(cal_param.T)
    print(cal_param.stereo_transform)

    left_params = cal_param.left_cam
    right_params = cal_param.right_cam

    print(left_params.fx,left_params.fy,left_params.cx,left_params.cy)
    print(right_params.fx,right_params.fy,right_params.cx,right_params.cy)


if __name__=='__main__':
    print(extract_params("~/genesis_eye/videos/HD720_SN27165053_16-29-04.svo"))

import pyzed.sl as sl
import cv2
import numpy as np



def main():
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
    main()

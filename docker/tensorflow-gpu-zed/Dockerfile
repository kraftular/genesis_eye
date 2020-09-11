ARG UBUNTU_RELEASE_YEAR
ARG CUDA_MAJOR
ARG CUDA_MINOR

FROM nvidia/cuda:${CUDA_MAJOR}.${CUDA_MINOR}-devel-ubuntu${UBUNTU_RELEASE_YEAR}.04

ARG UBUNTU_RELEASE_YEAR
ARG CUDA_MAJOR
ARG CUDA_MINOR
ARG ZED_SDK_MAJOR
ARG ZED_SDK_MINOR

ENV NVIDIA_DRIVER_CAPABILITIES \
        ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,video,utility

ENV TZ=US/Eastern
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# Setup the ZED SDK
RUN apt-get update -y && apt-get install --no-install-recommends lsb-release wget less udev sudo  build-essential cmake -y && \
    wget -O ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run https://download.stereolabs.com/zedsdk/${ZED_SDK_MAJOR}.${ZED_SDK_MINOR}/cu${CUDA_MAJOR}${CUDA_MINOR}/ubuntu${UBUNTU_RELEASE_YEAR} && \
    chmod +x ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run ; ./ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run -- silent skip_tools && \
    rm ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run && \
    rm -rf /var/lib/apt/lists/*

ADD get_python_api.py /get_python_api.py
ADD tensorflow-2.3.0-cp36-cp36m-linux_x86_64.whl /tensorflow-2.3.0-cp36-cp36m-linux_x86_64.whl

RUN apt-get update -y && apt-get install ffmpeg libsm6 libxext6  -y #needed for opencv
RUN apt-get install python3-wheel python3-setuptools -y #needed for tensorflow pip install??? installing pip packages doesn't seem to work for this.
RUN apt-get install python3-dev python3-tk -y

#get git, check out stuff
RUN apt-get install git -y
RUN git clone https://github.com/kraftular/tf2-pose-estimation.git


# ZED Python API, tensorflow, my tf2-pose stuff, etc.
RUN apt-get update -y && apt-get install --no-install-recommends python3 python3-pip -y && \
    python3 /get_python_api.py && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy opencv-python tqdm tables tensorflow_hub *.whl --use-feature=2020-resolver && \
    cd /tf2-pose-estimation && python3 -m pip install . && cd / && \
    apt-get install hdf5-tools -y && \
    apt-get remove --purge git -y && \
    rm *.whl ; rm -rf /var/lib/apt/lists/* ; rm /get_python_api.py

RUN rm -rf /tf2-pose-estimation #remove sources for adk openpose library

RUN chmod -R 755 /usr/local/zed 

RUN useradd -m genesis && echo "genesis:genesis" | chpasswd

RUN chown -R genesis:genesis /usr/local/zed #zed's api depends on the user being able to modify this directory. ooh, so creative!

USER genesis

WORKDIR /home/genesis/


#apt-get install --no-install-recommends python3-dev -y && \ #pip package opencv-python tries to build opencv, seems broken
#    python3 -m pip install setuptools scikit-build && \
#apt-get install --no-install-recommends python3-opencv -y && \ #nope, this wants to install entire desktop environment
#!/bin/bash

#ADK: modified build script from stereolabs. theirs didn't work with cuda11, and
#screwed up opencv-python installation as well. I removed all options except for the
#ones I need and simplified the project directory structure. there is still cruft
#in here reltaing to ROS and other garbage. Ignore it.

#This script produces the main docker image for use with genesis vision. because it
#relies on pulling updates etc, it's likely to break someday. Therefore, it's important
#to save the generated binary image somewhere even though it's likely several gigabytes

set -x

push_images=false
build_latest_only_images=true

ubuntu_release_year=(
  18
)

cuda_version=(
  "11.0"
)

zed_major_versions=(
  3
)

zed_minor_versions=(
  2
)

docker_image_variant=(
  devel
)

pwd_path=$(pwd)

for ZED_SDK_MAJOR in "${zed_major_versions[@]}" ; do
    for ZED_SDK_MINOR in "${zed_minor_versions[@]}" ; do
        for CUDA_VERSION in "${cuda_version[@]}" ; do
            for UBUNTU_RELEASE_YEAR in "${ubuntu_release_year[@]}" ; do
                for IMAGE_VARIANT in "${docker_image_variant[@]}" ; do

                    if $build_latest_only_images; then
                        if [ ${ZED_SDK_MINOR} -ne ${zed_minor_versions[-1]} ] ; then
                            continue
                        fi
                    fi

                    CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -f 1 -d '.')
                    CUDA_MINOR_VERSION=$(echo $CUDA_VERSION | cut -f 2 -d '.')

                    if [ ${UBUNTU_RELEASE_YEAR} == "16" ] ; then
                        ROS_DISTRO_ARG="kinetic"
                    elif [ ${UBUNTU_RELEASE_YEAR} == "18" ] ; then
                        ROS_DISTRO_ARG="melodic"
                        
                        # Not compatible with CUDA <= 9
                        if [ ${CUDA_MAJOR_VERSION} -le "9" ] ; then
                            continue
                        fi
                    fi

                    TAG_VERSION="tf_2.3gpu_zed_${ZED_SDK_MAJOR}.${ZED_SDK_MINOR}-${IMAGE_VARIANT}-cuda${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}-ubuntu${UBUNTU_RELEASE_YEAR}.04"
		    #IMAGE_PATH="${ZED_SDK_MAJOR}.X/ubuntu/${IMAGE_VARIANT}"
		    DIR_OF_SCRIPT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
		    IMAGE_PATH="${DIR_OF_SCRIPT}"
		    
                    cd "${IMAGE_PATH}"

                    echo "Building 'stereolabs/zed:${TAG_VERSION}'"

                    docker build --build-arg UBUNTU_RELEASE_YEAR=${UBUNTU_RELEASE_YEAR} \
                        --build-arg ZED_SDK_MAJOR=${ZED_SDK_MAJOR} \
                        --build-arg ZED_SDK_MINOR=${ZED_SDK_MINOR} \
                        --build-arg ROS_DISTRO_ARG=${ROS_DISTRO_ARG} \
                        --build-arg CUDA_MAJOR=${CUDA_MAJOR_VERSION} \
                        --build-arg CUDA_MINOR=${CUDA_MINOR_VERSION} \
                        -t "genesis/tfzed:${TAG_VERSION}" .

                    cd "${pwd_path}"
                    
                done
            done
        done
    done
done

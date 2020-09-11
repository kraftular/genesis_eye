import os
import platform
import sys
import re
import urllib.request

ZED_SDK_MAJOR = ""
ZED_SDK_MINOR = ""

CUDA_STR = ""

PYTHON_MAJOR = ""
PYTHON_MINOR = ""

OS_VERSION = ""
ARCH_VERSION = platform.machine()

whl_platform_str = ""

cuda_path = "/usr/local/cuda"
base_URL = "https://download.stereolabs.com/zedsdk/"

def check_valid_file(file_path):
    file_size = os.stat(file_path).st_size / 1000.
    # size > 150 Ko
    return (file_size > 150)

def check_cuda_version(cuda_path_version):
    global CUDA_STR
    #with open(cuda_path_version, "r", encoding="utf-8") as myfile:
    #    data = myfile.read()
    #    p = re.compile("CUDA Version (.*)")
    #    CUDA_VERSION = p.search(data).group(1)
    #    temp = re.findall(r'\d+', CUDA_VERSION) 
    #    res = list(map(int, temp)) 
    CUDA_MAJOR = 11#int(res[0])
    CUDA_MINOR = 0#int(res[1])
    print("CUDA " + str(CUDA_MAJOR) + "." + str(CUDA_MINOR))
    CUDA_STR = "cu" + str(CUDA_MAJOR) + str(CUDA_MINOR)

def check_zed_sdk_version_private(file_path):
    global ZED_SDK_MAJOR
    global ZED_SDK_MINOR
    with open(file_path, "r", encoding="utf-8") as myfile:
        data = myfile.read()

    p = re.compile("ZED_SDK_MAJOR_VERSION (.*)")
    ZED_SDK_MAJOR = p.search(data).group(1)

    p = re.compile("ZED_SDK_MINOR_VERSION (.*)")
    ZED_SDK_MINOR = p.search(data).group(1)

def check_zed_sdk_version(file_path):
    file_path_ = file_path+"/sl/Camera.hpp"
    try:
        check_zed_sdk_version_private(file_path_)
    except AttributeError:
        file_path_ = file_path+"/sl_zed/defines.hpp"
        check_zed_sdk_version_private(file_path_)
    
if sys.platform == "win32":
    if os.getenv("ZED_SDK_ROOT_DIR") is None:
        print(" you must install the ZED SDK.")
        exit(1)
    elif os.getenv("CUDA_PATH") is None:
        print("Error: you must install Cuda.")
        exit(1)
    else:
        check_zed_sdk_version(os.getenv("ZED_SDK_ROOT_DIR")+"/include")
        cuda_path_version = os.getenv("CUDA_PATH") + "/version.txt"
    OS_VERSION = "win"
    check_cuda_version(cuda_path_version)
    whl_platform_str = "win"

elif "linux" in sys.platform:

    if "aarch64" in ARCH_VERSION:
        with open("/etc/nv_tegra_release", "r", encoding="utf-8") as myfile:
            data = myfile.read()
        number_extraction = re.findall(r'\d+', data)
        TEGRA_RELEASE_MAJOR = int(number_extraction[0])
        TEGRA_RELEASE_MINOR = int(number_extraction[1])
        #TEGRA_RELEASE_PATCH = int(number_extraction[2])

        #TEGRA_RELEASE = str(TEGRA_RELEASE_MAJOR) + "." + str(TEGRA_RELEASE_MINOR) + "." + str(TEGRA_RELEASE_PATCH)
        #print(TEGRA_RELEASE)

        if TEGRA_RELEASE_MAJOR < 32:
            print('Unsupported jetpack version')
            exit(1)
        elif TEGRA_RELEASE_MAJOR == 32:
            if TEGRA_RELEASE_MINOR == 2:
                JETSON_JETPACK="42"
            elif TEGRA_RELEASE_MINOR == 3:
                JETSON_JETPACK="43"
            elif TEGRA_RELEASE_MINOR == 4:
                JETSON_JETPACK="44"
            else:
                print('Unsupported jetpack version')
                exit(1)
        else:
            print('Unsupported jetpack version')
            exit(1)
        print("JETPACK " + str(JETSON_JETPACK))
        CUDA_STR = "jp" + JETSON_JETPACK
        OS_VERSION = "jetsons"
    else:
        with open("/etc/lsb-release", "r", encoding="utf-8") as myfile:
            data = myfile.read()
        p = re.compile("DISTRIB_RELEASE=(.*)")
        DISTRIB_RELEASE = p.search(data).group(1).split(".")[0]
        p = re.compile("DISTRIB_ID=(.*)")
        DISTRIB_ID = p.search(data).group(1).lower()
        OS_VERSION = DISTRIB_ID + DISTRIB_RELEASE

        if not os.path.isdir(cuda_path):
            print("Error: you must install Cuda.")
            exit(1) 
        cuda_path_version = cuda_path + "/version.txt"
        check_cuda_version(cuda_path_version)

    zed_path = "/usr/local/zed"
    if not os.path.isdir(zed_path):
        print("Error: you must install the ZED SDK.")
        exit(1)
    check_zed_sdk_version(zed_path+"/include")
    whl_platform_str = "linux"
else:
    print ("Unknown system.platform: %s  Installation failed, see setup.py." % sys.platform)
    exit(1)    

PYTHON_MAJOR = platform.python_version().split(".")[0]
PYTHON_MINOR = platform.python_version().split(".")[1]

whl_python_version = "-cp" + str(PYTHON_MAJOR) + str(PYTHON_MINOR) + "-cp" + str(PYTHON_MAJOR) + str(PYTHON_MINOR)
if int(PYTHON_MINOR) < 8 :
    whl_python_version += "m"

print("Platform " + str(OS_VERSION))
print("ZED " + str(ZED_SDK_MAJOR) + "." + str(ZED_SDK_MINOR))
print("Python " + str(PYTHON_MAJOR) + "." + str(PYTHON_MINOR))

whl_file_URL = base_URL + str(ZED_SDK_MAJOR) + "." + str(ZED_SDK_MINOR) + "/" + OS_VERSION + "/" + CUDA_STR + "/py" + str(PYTHON_MAJOR) + str(PYTHON_MINOR)
print("Downloading python package from " + whl_file_URL + " ...")
whl_file = "pyzed-" + str(ZED_SDK_MAJOR) + "." + str(ZED_SDK_MINOR) + whl_python_version + "-" + whl_platform_str + "_" + str(ARCH_VERSION).lower() + ".whl"
urllib.request.urlretrieve(whl_file_URL, whl_file)
# Warning doesn't handle missing remote file yet and will probably download an html

if check_valid_file(whl_file):
    print("\n-> Please make sure numpy is installed \n python3 -m pip install cython\n python3 -m pip install numpy")
    print("File saved into " + whl_file)
    print("To install it run : \n python3 -m pip install "+ whl_file)
else:
    print("\nUnsupported platforms, no pyzed file available for this configuration\n It can be manually installed from source https://github.com/stereolabs/zed-python-api")

import subprocess
import logging
import argparse
from tqdm import tqdm
import os

logging.basicConfig(level=logging.DEBUG)

processing_class_names =[
    'SVOExtractImages',
    'DetectionCoCo',
    'GlobalSegmentation',
    'RefineDetections',
    ]

def process(classname,svo_list):
    import pipeline
    clazz = eval('pipeline.'+classname)
    instance = clazz()
    for svo in tqdm(svo_list):
        logging.info("processing %s"%svo)
        instance(svo)

def do_stages(stages,svos):
    me = os.path.abspath(__file__)
    for stage in tqdm(stages):
        retval = subprocess.call(
            "python3 {me} --stages {stage} --svo_files {files}".format(
                me=me,
                stage=stage,
                files=' '.join(svos)),
            shell=True)
        if retval!=0:
            return retval
    return 0

def main():
    parser = argparse.ArgumentParser(description='run the preprocessing pipeline')
    parser.add_argument('--svo_files', type=str ,dest='svos',nargs='+')
    parser.add_argument('--stages',
                        default=processing_class_names,
                        type=str ,dest='stages',
                        nargs = '+')
    args = parser.parse_args()
    if len(args.stages)==1:
        process(args.stages[0],args.svos)
        exit(0)
    elif len(args.stages)>1:
        exit(do_stages(args.stages,args.svos))
    else:
        raise ValueError(args.stages)


if __name__=='__main__':
    main()

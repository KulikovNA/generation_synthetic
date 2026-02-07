import argparse
import os
import shlex
import subprocess
import multiprocessing
import time

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpu', type=int, default=0, 
    help='Number of rendering processes to run in parallel. Must not exceed the number of available GPUs.'
)
parser.add_argument('bop_parent_path', nargs='?', default="/home/nikita/", help="Path to the bop datasets parent directory")
parser.add_argument('bop_dataset_name', nargs='?', default="databot", help="Main BOP dataset")
parser.add_argument('bop_toolkit_path', nargs='?', default="/home/nikita/BOP_tools/bop_toolkit", help="Path to bop toolkit")
parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', nargs='?', default="examples/bop_object_physics_positioning/output1", help="Path to where the final files will be saved ")
#parser.add_argument('runs', default=3, help="The number of times the objects should be repositioned and rendered using 2 to 5 random camera poses.")
args = parser.parse_args()

start_time = time.time()

# run BlenderProc `run.py` multiple times on a particular GPU
def env_one(process_idx, 
            parent_path=args.bop_parent_path,
            dataset_name=args.bop_dataset_name,
            toolkit_path=args.bop_toolkit_path,
            cc_tpath=args.cc_textures_path,
            output_dir=args.output_dir):
    
    #################################################
    #            основные переменные                #
    #################################################
    chunk = 2 #сколько всего будет чанков
    group_chunk = 1 #по сколько чанков будет генериться в подпроцессе 
    #c изменением переменных, обязательно менять аргументы в cmd
    runs = 9 #сколько нужно итераций, чтобы полностью заполнить чанки: (runs = 40) * group_chunk
    
    # set visible GPU
    env_1 = os.environ.copy()
    env_1['CUDA_VISIBLE_DEVICES'] = str(process_idx)
    
    # replace '%%' with process index in args
    #arg_string = ' '.join(args).replace('%%', str(process_idx))
    #arr_group = [i for i in range(len(group_chunk))]
    #print(arr_group)
    
    p1 = []
    
    j = [i for i in range(0, chunk)]
    l = [j[d:d+group_chunk] for d in range(0, len(j), group_chunk)]

    print(l)

    for i in range(len(l)):
        #здесь конечные элементы - изменить на равное колличество group_chunk
        #cmd = f'blenderproc run examples/datasets/bop_object_physics_positioning/main.py {parent_path} {dataset_name} {toolkit_path} {cc_tpath} {output_dir} {runs} {l[i][0]}'
        cmd = f'blenderproc run examples/advanced/coco_annotation/main.py {parent_path} {dataset_name} {toolkit_path} {cc_tpath} {output_dir}'
        p1.append(multiprocessing.Process(target = run_proc, args = (cmd, env_1,)))
        p1[i].start()
    

def run_proc(cmd, env):
    subprocess.run(cmd, env=env, shell=True, check=True)

def create_mask(dataset_name, toolkit_path, output_dir):
    cmd_musk = f'python {toolkit_path}/scripts/create_mask.py {dataset_name} {output_dir}'
    subprocess.run(cmd_musk, shell=True, check=True)

def create_gt_info(dataset_name, toolkit_path, output_dir):
    cmd_gt_info = f'python {toolkit_path}/scripts/create_gt_info.py {dataset_name} {output_dir}'
    subprocess.run(cmd_gt_info, shell=True, check=True)

# launch runners assigned to each GPU
if __name__ == '__main__':
    proc = args.gpu
    
    dataset_name=args.bop_dataset_name
    toolkit_path=args.bop_toolkit_path
    output_dir=args.output_dir

    p1 = multiprocessing.Process(target=env_one, args=(proc,))
    p1.start()
    p1.join()

    #создаем маски
    mask = multiprocessing.Process(target=create_mask, args=(dataset_name, toolkit_path, output_dir,))
    mask.start()
    mask.join()
    #делаем задержку
    time.sleep(120)
    #создаем gt_info 
    gt = multiprocessing.Process(target=create_gt_info, args=(dataset_name, toolkit_path, output_dir,))
    gt.start()
    gt.join()
    print('время работы программы: {}'.format(time.time() - start_time))

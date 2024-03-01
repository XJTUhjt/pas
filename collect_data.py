import logging
import os
import sys
from matplotlib import pyplot as plt
import torch
import shutil


from rl.vec_env.envs import make_vec_envs
from collecting_step import CollectingStep
from crowd_sim import *
from arguments import get_args
from crowd_nav.configs.config import Config

###########
# Things to change in config
# robot.policy = "orca' 
# sim.collectingdata = True
# sim.human_num = 6

# Things to change in arguments
# 'VAEdata/train' or 'VAEdata/val' or 'VAEdata/test'
#############

def main():
    data_args = get_args()

    # save policy to output_dir
    if os.path.exists(data_args.output_dir) and data_args.overwrite: # if I want to overwrite the directory
        shutil.rmtree(data_args.output_dir)  # delete an entire directory tree

    if not os.path.exists(data_args.output_dir):
        os.makedirs(data_args.output_dir)


    config = Config()
    data_args = get_args()

    # configure logging and device
    # print data result in log file
    log_file = os.path.join(data_args.output_dir,'data')
    if not os.path.exists(log_file):
        os.mkdir(log_file)
    log_file = os.path.join(data_args.output_dir, 'data_visual.log')


    #logging配置
    file_handler = logging.FileHandler(log_file, mode='w') #建了一个FileHandler，它用于将日志记录写入文件。log_file是指定的日志文件名，mode='w'表示在写入日志之前清除现有文件内容
    stdout_handler = logging.StreamHandler(sys.stdout) #创建了一个StreamHandler，它用于将日志记录输出到标准输出（控制台）。这意味着日志消息将显示在终端窗口中
    level = logging.INFO #这一行设置了日志的记录级别为INFO，这意味着只有INFO级别或更高级别的日志消息将被记录
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],       #这一行用于配置logging模块的基本设置。它指定了记录级别、处理程序（handlers）、日志消息的格式和日期时间格式
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    logging.info('robot FOV %f', config.robot.FOV)
    logging.info('humans FOV %f', config.humans.FOV)
    #这两行使用logging.info()方法记录INFO级别的日志消息。
    #它们记录了有关机器人和人类FOV（视野）的信息，其中config.robot.FOV和config.humans.FOV是相关数据的值。日志消息会根据前面设置的格式写入到标准输出和日志文件


    #cuda配置
    torch.manual_seed(data_args.seed)
    torch.cuda.manual_seed_all(data_args.seed)
    if data_args.cuda:
        if data_args.cuda_deterministic:
            # reproducible but slower
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            # not reproducible but faster
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False


    torch.set_num_threads(1)
    device = torch.device("cuda" if data_args.cuda else "cpu")

    logging.info('Create other envs with new settings')


    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    ax.set_xlabel('x(m)', fontsize=16)
    ax.set_ylabel('y(m)', fontsize=16)


    env_name = data_args.env_name
    phase = data_args.output_dir.split('/')[1]
    print('phase', phase)    

    #! collectingtraindata : When collecting test data, this should be false.
    envs = make_vec_envs(env_name, data_args.seed, 1,
                            data_args.gamma, device, allow_early_resets=True,
                            envConfig=config, ax=ax, phase=phase)

    #设置不同数据集sample数量
    if phase =='train':
        data_size = 1000  # 500 for turtlebot exp w/ 4 humans
    elif phase =='val':
        data_size = 100 # 50 for turtlebot exp w/ 4 humans
    elif phase =='test':
        data_size = 200 # 100 for turtlebot exp w/ 4 humans

    else:
        raise NotImplementedError


    visualize = True
    
    #开始数据采集
    CollectingStep(data_args, config, data_args.output_dir, envs,  device, data_size, logging, visualize)


if __name__ == '__main__':
    main()

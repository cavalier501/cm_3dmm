# -*- coding: utf-8 -*-
# @Time    : 2021/9/1 18:30
# @Author  : xinyuan tu
# @File    : utils.py
# @Software: PyCharm

import torch
import torch.nn as nn
import os
import time
import math
import numpy as np
import torch
import logging
#from lib.render import *

def load_state_dict(module,state_dict):


    def _load_state_dict_into_module(state_dict, module, strict=True):
        own_state = module.state_dict()
        count = 0

        for name, param in state_dict.items():
            #name = name.replace('.module','')
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].resize_as_(param)
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))


    _load_state_dict_into_module(state_dict,module)

def dict_to_cuda(input_info):
    for key, value in input_info.items():
        if key=='basename':
            continue
        if type(value) is np.ndarray:
            value=torch.from_numpy(value);
        if type(value) is int or type(value) is float:
            value=torch.tensor(value)
        if value.dtype==torch.double:
            value=value.float()
        if type(value) is torch.Tensor:
            input_info[key] = value.cuda(non_blocking=False)
    return input_info


def del_cuda(input_info,loss,output):
    for key, value in input_info.items():
        if key=='basename':
            continue
        del value

    for key, value in loss.items():
        if key=='basename':
            continue
        del value
    for key, value in output.items():
        if key=='basename':
            continue
        del value

def make_dirs(arg):
    t = time.gmtime()
    event_path = os.path.join(arg.event_path,arg.state, str(t.tm_mon).rjust(2, '0') + str(t.tm_mday).rjust(2, '0') + str((t.tm_hour+8)%24).rjust(2, '0') + str((t.tm_min)%60).rjust(2, '0'))
    while(os.path.isdir(event_path)):
        event_path=event_path+"_2"
    if not os.path.isdir(event_path):
        os.makedirs(event_path)
    checkpoint_path = os.path.join(event_path, 'checkpoint_path')
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    summary_path = os.path.join(event_path,'summary_path')
    if not os.path.isdir(summary_path):
        os.makedirs(summary_path)

    # for visualize
    if arg.test_save:
        output_path = os.path.join(event_path, 'output')
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
    return event_path


def save_obj_pts(filename, vertices, faces):
    import os
    if np.min(faces)==0:
        faces=faces+1
    with open(filename, 'w') as f:
        f.write('# %s\n' % os.path.basename(filename))
        f.write('#\n')
        f.write('\n')
        for vertex in vertices:
            f.write('v %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2]))
        f.write('\n');
        for face in faces:
            f.write('f %d %d %d\n' % (face[0], face[1], face[2]));
        f.write('\n');


def test_dict_to_cuda():
    test_1=np.zeros([3,3],dtype=np.uint8);
    test_2=np.zeros([3,3],dtype=np.float);
    test_3=torch.zeros([3,3],dtype=torch.int8);
    test_4=torch.zeros([3,3],dtype=torch.float32,requires_grad=False);
    example_dict={
        "test_1":test_1,
        "test_2":test_2,
        "test_3":test_3,
        "test_4":test_4
    }
    # print(type(test_1));
    # torch_int_list=[torch.int,torch.int8,torch.int16,torch.int32]
    # print(test_3.dtype in torch_int_list)
    out_dict=dict_to_cuda(example_dict);
    print(out_dict)
    return

# 关于logging的设置
# 这里也统一放在了utils.py中
import logging
import os
import time
from configs.configuration import *

modelconfig = args
def configure_logging(event_path,txt_path:str='checkpoint_path'):
    root_logger = logging.getLogger("")
    root_logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    root_logger.addHandler(console)
    log_filename = os.path.join(os.path.join(event_path,txt_path),'log.txt')
    logbook = logging.FileHandler(filename=log_filename, mode="a", encoding="utf-8")
    logbook.setLevel(logging.INFO)
    root_logger.addHandler(logbook)
    return root_logger

def setup_logging_and_parse_arguments(logger):
    for argument, value in sorted(vars(modelconfig).items()):
        logger.info('{}: {}'.format( argument, value))




def main():
    test_dict_to_cuda();
    return
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']="3"
    main()

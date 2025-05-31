import argparse
import numpy as np
import math

parse = argparse.ArgumentParser('configration of program')
parse.add_argument('--CUDA_VISIBLE_DEVICES',type=str,default="1")


logger_parse = parse.add_argument_group("logger")
logger_parse.add_argument('--is_logger',type=bool,default=True)
logger_parse.add_argument('--event_path',type=str,default='./experiment')
logger_parse.add_argument('--iswriter',type=bool,default=False)


model_parse = parse.add_argument_group("model")
model_parse.add_argument("--warp_type",type=str,default="translation")

train_parse = parse.add_argument_group("train")
train_parse.add_argument('--batch_size',type=int,default=20)
train_parse.add_argument('--N_vert_for_train',type=int,default=10000)
train_parse.add_argument('--N_continuous_for_train',type=int,default=5000)

# optimizer
train_parse.add_argument('--lr',type=float,default=2e-4)
train_parse.add_argument('--num_epochs',type=int,default=5000)
train_parse.add_argument('--writesummary',type=int,default=30)
# should be changed for different train stage
train_parse.add_argument('--pre_train', type=bool, default=False)
train_parse.add_argument('--pre_train_ckpt', type=str, default='')
train_parse.add_argument('--already_trained_epoch', type=int, default=0)

test_parse = parse.add_argument_group("test")
test_parse.add_argument('--batch_size_test',type=int,default=1)
test_parse.add_argument('--test_save',type=bool,default=True)
test_parse.add_argument('--N_vert_for_test',type=int,default=2000)
test_parse.add_argument('--N_continuous_for_test',type=int,default=1000)

args = parse.parse_args()
args.milestones=[2000+8000,4000+8000,6000+8000,]
# args.milestones=[int(args.num_epochs*0.25),int(args.num_epochs/2),int(args.num_epochs*0.75)]


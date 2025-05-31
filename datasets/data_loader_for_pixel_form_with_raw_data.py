import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
PREDEFINED_MODEL_DATA = os.path.join(PROJECT_ROOT, "predefined_model_data")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets") # 指明数据集的txt文件所在路径
DATASET_ROOT = ""        # 替换为txt文件中实际数据集的绝对路径前缀
"""
数据加载器的字典，包含与平均脸和原始数据上查询的UV坐标相关的几何位置信息。
example_dict={
    "bs": [bs],  # 批量大小
    "id_index":  [bs],  # 身份索引
    "exp_index": [bs],  # 表情索引
    "point_num": [bs],  # 点的数量         
    "point_uv":  [bs, N, 2], (0,1)  # 点的UV坐标
    "point_xyz_on_detail":       [bs, N, 3],  # 原始采集人脸上的三维坐标
    "point_xyz_on_average_face": [bs, N, 3],  # 平均脸上的三维坐标
}
"""


class data_loader_template_on_raw(Dataset):
    def __init__(self, args, phase="train"):
        assert phase in ["train", "test", "val"], "Invalid phase specified."
        self.phase = phase
        self.bs = args.batch_size
        self.N_vert = args.N_vert_for_train if phase == "train" else args.N_vert_for_test
        self.N_continuous = args.N_continuous_for_train if phase == "train" else args.N_continuous_for_test
        self.vert_average_face = torch.tensor(
            np.load(os.path.join(PREDEFINED_MODEL_DATA, "average_face_vert.npy"))
        )
        self.obj_list = []
        self.id_list = []
        self.exp_list = []

    def __len__(self):
        return len(self.obj_list)

    def __getitem__(self, item):
        index = item % len(self.obj_list)
        obj_name = self.obj_list[index]
        splitted_name = obj_name.split("/")
        id_name, exp_name = splitted_name[-2], splitted_name[-1]
        id_index = self.id_list.index(id_name)
        exp_index = self.exp_list.index(exp_name)

        # 加载离散语义顶点数据
        vert_data = torch.load(os.path.join(obj_name, "vert_data_full.pt"))
        vert_shuffle_order = torch.load(os.path.join(obj_name, "vert_shuffle_index_full.pt"))
        vert_indices = vert_shuffle_order[:self.N_vert]
        vert_xyz_on_detail = vert_data[vert_indices, 6:9]
        vert_xyz_on_average_face = self.vert_average_face[vert_indices]
        vert_point_uv = vert_data[vert_indices, 1:3]

        # 加载连续语义顶点数据
        continuous_data = torch.load(os.path.join(obj_name, "continuous_data_filtered_distance_0_3.pt"))
        continuous_shuffle_order = torch.load(
            os.path.join(obj_name, "continuous_data_filtered_distance_0_3_shuffle_index.pt")
        )
        continuous_indices = continuous_shuffle_order[:self.N_continuous]
        continuous_xyz_on_detail = continuous_data[continuous_indices, 14:17]
        continuous_xyz_on_average_face = torch.load(
            os.path.join(obj_name, "continuous_data_filtered_distance_0_3_xyz_on_average_face.pt")
        )[continuous_indices]
        continuous_point_uv = continuous_data[continuous_indices, 0:2]

        # 构建 example_dict
        example_dict = {
            "bs": self.bs,
            "id_index": id_index,
            "exp_index": exp_index,
            "point_num": len(vert_indices) + len(continuous_indices),
            "point_xyz_on_detail": torch.cat((vert_xyz_on_detail, continuous_xyz_on_detail), dim=0),
            "point_xyz_on_average_face": torch.cat((vert_xyz_on_average_face, continuous_xyz_on_average_face), dim=0),
            "point_uv": torch.cat((vert_point_uv, continuous_point_uv), dim=0),
        }
        return example_dict

    def load_data(self, dataset_list: str, face_num: int):
        self.obj_list = []
        with open(dataset_list, "r") as f:
            lines = [line.strip() for line in f if not line.startswith("#")]
            # 替换绝对路径为基于 DATASET_ROOT 的相对路径
            self.obj_list = [os.path.join(DATASET_ROOT, line) for line in lines[:face_num]]


class id_1_exp_1_dataset_loader(data_loader_template_on_raw):
    def __init__(self, args, phase: str = "train"):
        super().__init__(args, phase)
        self.id_list = ["1"]
        self.exp_list = ["1_neutral"]
        # 数据集文件
        dataset_list = os.path.join(DATASETS_DIR, "dataset_list_for_initial_version_id_1_exp_1.txt")
        self.load_data(dataset_list, face_num=1)

class id_1_exp_1_dataset_train(id_1_exp_1_dataset_loader):
    def __init__(self, args):
        super().__init__(args, phase="train")


class id_280_exp_20_dataset_loader(data_loader_template_on_raw):
    def __init__(self, args, phase: str = "train"):
        super().__init__(args, phase)
        self.id_list = [str(i) for i in range(1, 301)]
        self.exp_list = [
            "1_neutral", "2_smile", "3_mouth_stretch", "4_anger", "5_jaw_left", "6_jaw_right",
            "7_jaw_forward", "8_mouth_left", "9_mouth_right", "10_dimpler", "11_chin_raiser",
            "12_lip_puckerer", "13_lip_funneler", "14_sadness", "15_lip_roll", "16_grin",
            "17_cheek_blowing", "18_eye_closed", "19_brow_raiser", "20_brow_lower"
        ]
        # 数据集文件
        dataset_list = os.path.join(DATASETS_DIR, "dataset_list_for_initial_version_id_300_exp_20.txt")
        self.load_data(dataset_list, face_num=5595)

class id_280_exp_20_dataset_train(id_280_exp_20_dataset_loader):
    def __init__(self, args):
        super().__init__(args, phase="train")

class id_280_exp_20_dataset_test(id_280_exp_20_dataset_loader):
    def __init__(self, args):
        super().__init__(args, phase="test")

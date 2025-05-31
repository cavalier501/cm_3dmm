import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from PIL import Image
from tqdm import tqdm
from typing import Dict
from torch.autograd import Variable

# 添加当前项目根目录到路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..",)
sys.path.append(PROJECT_ROOT)

from model.deepsdf_like_mlp_model import deepsdf_like_network
import model.hash_encoding as hash_encoding

# 定义warp函数，避免导入问题
def warp(xyz, deformation, warp_type="translation"):
    """变形函数实现"""
    if warp_type == "translation":
        return xyz + deformation
    else:
        return xyz + deformation

def save_obj_pts(path, vertices, faces):
    """保存OBJ文件函数实现"""
    try:
        with open(path, 'w') as f:
            # 写入顶点
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            # 写入面片（OBJ格式索引从1开始）
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    except Exception as e:
        print(f"Error saving OBJ file {path}: {e}")

def positional_encoding(uv_coords, num_encoding_functions):
    """位置编码函数"""
    encoding_list = [uv_coords]
    for i in range(num_encoding_functions):
        for fn in [torch.sin, torch.cos]:
            encoding_list.append(fn((2 ** i) * np.pi * uv_coords))
    return torch.cat(encoding_list, dim=-1)

# 配置常量
PREDEFINED_MODEL_DATA = os.path.join(PROJECT_ROOT, "predefined_model_data")

class ModelConfig:
    """模型配置类"""
    def __init__(self):
        # 数据文件配置
        self.data_files = {
            'average_face': "average_face_vert.npy",
            'front_faces': "front_faces.npy",
            'vert_texcoords': "vert_texcoords.npy",
            'boundary_vert_id': "boundary_vert_id.npy",
            'boundary_vert_seg': "boundary_vert_seg.npy",
            'mask_full': "rasterize_predef/mask_full.jpg"
        }
        self.boundary_vert_num = 1197

class BaseUVEmbeddingModel(nn.Module):
    """基础UV嵌入模型类"""
    
    def __init__(self, args, id_num=280, exp_num=20, 
                 id_embedding_dim=128, exp_embedding_dim=128,
                 id_embedding_dropout_rate=0.0, exp_embedding_dropout_rate=0.0):
        super().__init__()
        self.config = ModelConfig()
        self.id_num = id_num
        self.exp_num = exp_num
        self.id_embedding_dim = id_embedding_dim
        self.exp_embedding_dim = exp_embedding_dim
        
        # 嵌入层初始化
        self._init_embeddings(id_embedding_dropout_rate, exp_embedding_dropout_rate)
        
        # 初始化基础数据
        self._load_predefined_data()

    def _init_embeddings(self, id_dropout_rate, exp_dropout_rate):
        """初始化嵌入层"""
        self.id_embeddings = nn.Embedding(
            num_embeddings=(self.id_num * self.exp_num),
            embedding_dim=self.id_embedding_dim            
        )
        self.id_embedding_dropout = nn.Dropout(p=id_dropout_rate)
        
        self.exp_embeddings = nn.Embedding(
            num_embeddings=(self.id_num * self.exp_num),
            embedding_dim=self.exp_embedding_dim            
        )       
        self.exp_embedding_dropout = nn.Dropout(p=exp_dropout_rate)
        
        # 权重初始化
        nn.init.normal_(self.id_embeddings.weight, mean=0, std=0.01)
        nn.init.normal_(self.exp_embeddings.weight, mean=0, std=0.01)

    def _load_predefined_data(self):
        """加载预定义数据"""
        # 加载基础几何数据
        self.average_face = torch.tensor(
            np.load(os.path.join(PREDEFINED_MODEL_DATA, self.config.data_files['average_face']))
        )
        
        tri_data = np.load(os.path.join(PREDEFINED_MODEL_DATA, self.config.data_files['front_faces']))
        if np.min(tri_data) == 1:
            tri_data = tri_data - 1
        self.tri_tu = torch.tensor(tri_data)
        
        self.vert_texcoords = torch.tensor(
            np.load(os.path.join(PREDEFINED_MODEL_DATA, self.config.data_files['vert_texcoords']))
        )
        
        # 加载边界数据
        self.boundary_vert_id = torch.tensor(
            np.load(os.path.join(PREDEFINED_MODEL_DATA, self.config.data_files['boundary_vert_id']))
        )
        self.boundary_vert_seg = torch.tensor(
            np.load(os.path.join(PREDEFINED_MODEL_DATA, self.config.data_files['boundary_vert_seg']))
        )
        self.boundary_vert_num = self.config.boundary_vert_num
        
        # 加载邻域数据
        self._load_neighborhood_data()
        
        # 加载其他预定义数据
        self._load_additional_data()

    def _load_neighborhood_data(self):
        """加载邻域数据"""
        neighbor_files = [
            ('order_1_vert_pair_list.pkl', 37026),
            ('order_2_vert_pair_list.pkl', 73629),
            ('order_3_vert_pair_list.pkl', 110139)
        ]
        
        for i, (filename, size) in enumerate(neighbor_files, 1):
            with open(os.path.join(PREDEFINED_MODEL_DATA, filename), 'rb') as file:
                neighborhood_list = pickle.load(file)
            
            setattr(self, f'order_{i}_vert_neighborhood_list', 
                   torch.tensor(neighborhood_list, dtype=torch.long))
            setattr(self, f'order_{i}_similar_target', torch.ones(size))

    def _load_additional_data(self):
        """加载额外的预定义数据"""
        # 加载ID/EXP相似性标签数据
        self._setup_similarity_labels()
        
        
        # 加载上采样模板数据
        self._load_upsampled_data()
        
        # 加载UV mask
        self._load_uv_mask()

    def _load_upsampled_data(self):
        """加载上采样模板数据"""
        try:
            subdivision_path = os.path.join(PREDEFINED_MODEL_DATA, "subdivision_template")
            
            self.upsampled_point_vert_id = torch.tensor(
                np.load(os.path.join(subdivision_path, "sub_division_point_vert_id.npy"))
            ).to(torch.int)
            
            self.upsampled_point_bc = torch.tensor(
                np.load(os.path.join(subdivision_path, "sub_division_point_bc.npy"))
            ).to(torch.float)
            
            self.upsampled_point_xyz_on_average_face = torch.tensor(
                np.load(os.path.join(subdivision_path, "sub_division_average_face.npy"))
            ).to(torch.float)
            
            self.upsampled_tri = torch.tensor(
                np.load(os.path.join(subdivision_path, "sub_division_tri.npy"))
            )
            
            # Edge 2 subdivision data
            edge2_path = os.path.join(PREDEFINED_MODEL_DATA, "subdivision_template_edge_2")
            self.upsampled_tri_edge_2 = torch.tensor(
                np.load(os.path.join(edge2_path, "sub_division_tri.npy"))
            )
            self.upsampled_point_uv_edge_2 = torch.tensor(
                np.load(os.path.join(edge2_path, "sub_division_point_uv.npy"))
            ).to(torch.float32)
        except FileNotFoundError:
            print("Warning: Some upsampled data files not found, skipping...")
            pass

    def _load_uv_mask(self):
        """加载UV mask"""
        try:
            mask_path = os.path.join(PREDEFINED_MODEL_DATA, self.config.data_files['mask_full'])
            mask_PIL = Image.open(mask_path)
            
            uv_size = getattr(self, 'uv_pixel_size', 128) # 使用 getattr 获取 uv_pixel_size
            mask_PIL = mask_PIL.resize((uv_size, uv_size))
            
            mask = torch.tensor(np.array(mask_PIL)).to(torch.float32)
            mask = mask / 255
            mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0), torch.ones(1,1,3,3)/9, padding=1).squeeze()
            
            mask = (mask != 0).float()
            
            mask_new = torch.zeros_like(mask)
            for i in range(uv_size):
                for j in range(uv_size):  
                    mask_new[i,j] = mask[uv_size-1-i,j]
                    
            self.uv_mask = mask_new.reshape(1, uv_size, uv_size)
        except FileNotFoundError:
            print("Warning: UV mask file not found, skipping...")
            pass

    def _setup_similarity_labels(self):
        """设置相似性标签"""
        id_embedding_same_label = []
        for i in range(self.id_num):
            for j in range(1, self.exp_num):
                id_embedding_same_label.append([i*self.exp_num, i*self.exp_num+j])
        self.id_embedding_same_label = torch.tensor(id_embedding_same_label)
        self.id_embedding_same_target = torch.ones(self.id_num*(self.exp_num-1))
        
        exp_embedding_same_label = []
        for j in range(self.exp_num):
            for i in range(1, self.id_num):
                exp_embedding_same_label.append([j, j+i*self.exp_num])
        self.exp_embedding_same_label = torch.tensor(exp_embedding_same_label)
        self.exp_embedding_same_target = torch.ones(self.exp_num*(self.id_num-1))
        
        id_combinations = self._get_combinations(self.id_num)
        exp_combinations = self._get_combinations(self.exp_num)
        
        id_embedding_unsame_label = []
        for combo in id_combinations:
            id_embedding_unsame_label.append([combo[0]*self.exp_num, combo[1]*self.exp_num])
        self.id_embedding_unsame_label = torch.tensor(id_embedding_unsame_label)
        self.id_embedding_unsame_target = 1 - torch.ones(self.id_num*(self.id_num-1)//2)
        
        exp_embedding_unsame_label = []
        for combo in exp_combinations:
            exp_embedding_unsame_label.append([combo[0], combo[1]])
        self.exp_embedding_unsame_label = torch.tensor(exp_embedding_unsame_label)
        self.exp_embedding_unsame_target = 1 - torch.ones(self.exp_num*(self.exp_num-1)//2)

    def _get_combinations(self, n):
        """生成组合"""
        return [[i, j] for i in range(n) for j in range(i + 1, n)]

    def basic_init_data_to_cuda(self):
        """将数据移动到CUDA设备"""
        model_device = next(self.parameters()).device
        
        data_attrs = [
            'average_face', 'tri_tu', 'vert_texcoords', 
            'boundary_vert_id', 'boundary_vert_seg',
            'continuous_data_point_vert_id', 'continuous_data_point_bc',
            'exp_embedding_same_label', 'exp_embedding_same_target',
            'id_embedding_same_label', 'id_embedding_same_target',
            'id_embedding_unsame_label', 'id_embedding_unsame_target',
            'exp_embedding_unsame_label', 'exp_embedding_unsame_target',
            'upsampled_point_vert_id', 'upsampled_point_bc', 
            'upsampled_point_xyz_on_average_face', 'upsampled_tri',
            'upsampled_tri_edge_2', 'upsampled_point_uv_edge_2', 'uv_mask'
        ]
        
        for i in range(1, 4):
            data_attrs.append(f'order_{i}_vert_neighborhood_list')
            data_attrs.append(f'order_{i}_similar_target')

        for attr in data_attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None: # 检查属性是否存在且不为None
                setattr(self, attr, getattr(self, attr).to(model_device))

    def id_label_similarity_loss_for_id_embeddings(self):
        """ID标签相似性损失"""
        id_0_same_embedding = self.id_embeddings(self.id_embedding_same_label[:,0])
        id_1_same_embedding = self.id_embeddings(self.id_embedding_same_label[:,1])
        id_similarity_loss = F.cosine_embedding_loss(
            id_0_same_embedding, id_1_same_embedding, 
            self.id_embedding_same_target, reduction="mean"
        )

        id_0_unsame_embedding = self.id_embeddings(self.id_embedding_unsame_label[:,0])
        id_1_unsame_embedding = self.id_embeddings(self.id_embedding_unsame_label[:,1])
        id_unsimilarity_loss = F.cosine_embedding_loss(
            id_0_unsame_embedding, id_1_unsame_embedding, 
            self.id_embedding_unsame_target, reduction="mean"
        )
        return (id_similarity_loss + id_unsimilarity_loss) / 2

    def exp_label_similarity_loss_for_exp_embeddings(self):
        """EXP标签相似性损失"""
        exp_0_same_embedding = self.exp_embeddings(self.exp_embedding_same_label[:,0])
        exp_1_same_embedding = self.exp_embeddings(self.exp_embedding_same_label[:,1])
        exp_similarity_loss = F.cosine_embedding_loss(
            exp_0_same_embedding, exp_1_same_embedding, 
            self.exp_embedding_same_target, reduction="mean"
        )

        exp_0_unsame_embedding = self.exp_embeddings(self.exp_embedding_unsame_label[:,0])
        exp_1_unsame_embedding = self.exp_embeddings(self.exp_embedding_unsame_label[:,1])
        exp_unsimilarity_loss = F.cosine_embedding_loss(
            exp_0_unsame_embedding, exp_1_unsame_embedding, 
            self.exp_embedding_unsame_target, reduction="mean"
        )
        return (exp_similarity_loss + exp_unsimilarity_loss) / 2
    
    def id_embedding_mean(self):
        id_embedding_mean = torch.mean(self.id_embeddings.weight, dim=0)
        id_embedding_mean_loss = torch.norm(id_embedding_mean)
        return id_embedding_mean, id_embedding_mean_loss
    
    def id_embedding_std(self, target_std=0.01):
        id_embedding_std = torch.std(self.id_embeddings.weight, dim=0)
        std = torch.ones_like(id_embedding_std) * target_std
        id_embedding_std_loss = torch.norm(id_embedding_std - std)
        return id_embedding_std, id_embedding_std_loss

    def exp_embedding_mean(self):
        exp_embedding_mean = torch.mean(self.exp_embeddings.weight, dim=0)
        exp_embedding_mean_loss = torch.norm(exp_embedding_mean)
        return exp_embedding_mean, exp_embedding_mean_loss

    def exp_embedding_std(self, target_std=0.01):
        exp_embedding_std = torch.std(self.exp_embeddings.weight, dim=0)
        std = torch.ones_like(exp_embedding_std) * target_std
        exp_embedding_std_loss = torch.norm(exp_embedding_std - std)
        return exp_embedding_std, exp_embedding_std_loss

    def mesh_save(self, tu_vert: torch.tensor, save_path: str):
        """保存网格"""
        self.basic_init_data_to_cuda()   
        save_obj_pts(
            save_path,
            vertices=tu_vert.cpu().numpy(),
            faces=self.tri_tu.cpu().numpy(),
        )

    def point_distance_loss(self, input_dict, predict_deform_dict):
        """计算点距离损失"""
        point_xyz_on_detail = input_dict["point_xyz_on_detail"].reshape(-1, 3)
        point_xyz_on_average_face = input_dict["point_xyz_on_average_face"].reshape(-1, 3)
        predict_deform = predict_deform_dict["predict_deform"].reshape(-1, 3)
        
        predict_xyz_on_detail = warp(
            xyz=point_xyz_on_average_face,
            deformation=predict_deform,
            warp_type="translation",
        ).squeeze(0)
        
        return torch.mean(
            torch.norm((point_xyz_on_detail - predict_xyz_on_detail), dim=1)
        )

    def init_weights(self, m, mean=0.0, std=0.01):
        """初始化权重"""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=mean, std=std)
            nn.init.constant_(m.bias, 0.0)

class pixel_uv_embedding_ngp(BaseUVEmbeddingModel):
    """NGP版本的像素UV嵌入模型"""
    
    def __init__(self, args, id_num=280, exp_num=20,
                 id_embedding_dim=128, exp_embedding_dim=128,
                 id_embedding_dropout_rate=0.0, exp_embedding_dropout_rate=0.0,
                 uv_pixel_size=128, hash_encoding_n_levels=16,
                 hash_encoding_n_features_per_level=16,
                 hash_encoding_base_resolution=1,
                 hash_encoding_finest_resolution=128,
                 nonlinearity="tanh_kx", tanh_kx=3,
                 surface_processor_deepsdf_like_network_mlp_out_dim_list=[512,256,256,128,32],
                 surface_processor_deepsdf_like_network_use_weight_norm=True,
                 surface_processor_deepsdf_like_network_dropout_rate=0.0,
                 scale=False):
        
        super().__init__(args, id_num, exp_num, id_embedding_dim, exp_embedding_dim,
                        id_embedding_dropout_rate, exp_embedding_dropout_rate)
        
        self.uv_pixel_size = uv_pixel_size
        self.scale = scale
        
        # 初始化UV哈希编码器 - 注意这里的命名更改
        self.uv_hash_enncoder = hash_encoding.MultiResHashGrid( # 更名为 uv_hash_enncoder
            2,
            n_levels=hash_encoding_n_levels,
            n_features_per_level=hash_encoding_n_features_per_level,
            base_resolution=hash_encoding_base_resolution,
            finest_resolution=hash_encoding_finest_resolution,
        )
        
        # 初始化表面处理器
        self.surface_processor = deepsdf_like_network(
            item_embedding_dim=self.id_embedding_dim + self.exp_embedding_dim,
            uv_dim=self.uv_hash_enncoder.output_dim, # 确保这里也使用更改后的名称
            mlp_out_dim_list=surface_processor_deepsdf_like_network_mlp_out_dim_list,
            nonlinearity=nonlinearity,
            tanh_k=tanh_kx,
            use_weight_norm=surface_processor_deepsdf_like_network_use_weight_norm,
            dropout_rate=surface_processor_deepsdf_like_network_dropout_rate,
        )
        
        self.init_weights(self.surface_processor)

    def forward(self, input_dict: Dict[str, torch.tensor]):
        """前向传播"""
        id_index = input_dict["id_index"]
        exp_index = input_dict["exp_index"]
        point_uv = input_dict["point_uv"]

        # 从实际的批处理张量中获取批大小和点数
        # id_index 的形状是 (current_batch_size)
        # point_uv 的形状是 (current_batch_size, num_points_per_sample, 2)
        current_batch_size = id_index.shape[0]
        num_points_per_sample = point_uv.shape[1]
        
        # 获取ID和EXP嵌入
        combined_index = id_index * self.exp_num + exp_index # Shape: (current_batch_size)
        
        id_embedding = self.id_embeddings(combined_index) # Shape: (current_batch_size, id_embedding_dim)
        # 扩展以匹配 (current_batch_size, num_points_per_sample, id_embedding_dim)
        id_embedding = id_embedding.unsqueeze(1).expand(current_batch_size, num_points_per_sample, self.id_embedding_dim)
        id_embedding = self.id_embedding_dropout(id_embedding)
        
        exp_embedding = self.exp_embeddings(combined_index) # Shape: (current_batch_size, exp_embedding_dim)
        # 扩展以匹配 (current_batch_size, num_points_per_sample, exp_embedding_dim)
        exp_embedding = exp_embedding.unsqueeze(1).expand(current_batch_size, num_points_per_sample, self.exp_embedding_dim)
        exp_embedding = self.exp_embedding_dropout(exp_embedding)
        
        # UV嵌入 - 注意这里的命名更改
        uv_embedding_on_input = self.uv_hash_enncoder(point_uv) # 使用更改后的名称
        
        # 预测变形
        predict_deformation = self.surface_processor(
            torch.cat([id_embedding, exp_embedding], dim=-1),
            uv_embedding_on_input,
        ) # Shape: (current_batch_size, num_points_per_sample, 3)
        
        # 重塑为 (-1, 3) 以匹配损失函数的期望
        predict_deformation = predict_deformation.reshape(-1, 3)
        
        if self.scale:
            predict_deformation = predict_deformation * 3
            
        return {"predict_deform": predict_deformation}
        
    def fitting(self, uv_queries: torch.Tensor, delta_xyz: torch.Tensor, max_iter_num=1000, distance_threshold=0.01, use_mean_std_prior=False, target_std=0.01):
        model_device=next(self.parameters()).device   
        self.basic_init_data_to_cuda()
        bs        = 1
        point_num = uv_queries.shape[0]     
        uv_queries=uv_queries.unsqueeze(0).to(model_device)
        delta_xyz = delta_xyz.to(model_device)
        id_embedding=torch.zeros(size=(1,self.id_embedding_dim),dtype=torch.float).to(model_device).requires_grad_();
        exp_embedding=torch.zeros(size=(1,self.exp_embedding_dim),dtype=torch.float).to(model_device).requires_grad_();
        id_embedding  = Variable(id_embedding,requires_grad=True)
        exp_embedding = Variable(exp_embedding,requires_grad=True)
        optimizer=torch.optim.Adam(params=[id_embedding,exp_embedding],lr=0.003);
        for epoch in range(max_iter_num):
            optimizer.zero_grad()
            id_embedding_expanded  = id_embedding.unsqueeze(1).expand(bs,point_num,self.id_embedding_dim)
            exp_embedding_expanded = exp_embedding.unsqueeze(1).expand(bs,point_num,self.exp_embedding_dim)
            uv_embedding_on_input=self.uv_hash_enncoder(uv_queries)

            predict_deformation=self.surface_processor(
                torch.cat([id_embedding_expanded,exp_embedding_expanded],dim=-1),
                uv_embedding_on_input,
            ).reshape(-1,3)
            if self.scale:
                predict_deformation=predict_deformation*3
            # loss
            average_point_distance_loss=torch.mean(
                torch.norm((delta_xyz-predict_deformation),dim=1)
            ) 
            if use_mean_std_prior:
                id_std         = torch.std(id_embedding.squeeze())
                exp_std        = torch.std(exp_embedding.squeeze())
                id_piror_loss  = torch.norm(id_std-torch.ones_like(id_std)*target_std)
                exp_piror_loss = torch.norm(exp_std-torch.ones_like(exp_std)*target_std)
                total_loss=average_point_distance_loss+\
                    0.1*(id_piror_loss+exp_piror_loss)
            else:
                total_loss=average_point_distance_loss
            total_loss.backward();
            optimizer.step();                                              
        id_embedding    = id_embedding.detach()
        exp_embedding   = exp_embedding.detach()
        fitted_delta_tu = predict_deformation.detach()
        mean_distance=average_point_distance_loss.item()
        return id_embedding,exp_embedding,fitted_delta_tu,mean_distance



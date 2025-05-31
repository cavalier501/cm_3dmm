import os
import torch
from configs.configuration import *
from tqdm import tqdm
from utils.utils import *
import matplotlib.pyplot as plt
import pylab as pl
from torch.utils.data import DataLoader
from datasets import data_loader_for_pixel_form_with_raw_data
from model.pixel_uv_embedding_ngp import pixel_uv_embedding_ngp


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR,  "..")
PREDEFINED_MODEL_DATA = os.path.join(PROJECT_ROOT, "predefined_model_data")
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "pixel_uv_embedding", "experiment")


def single_face_fitting(
    detail_pts_path,
    ckpt_path,
    average_face_path=os.path.join(PREDEFINED_MODEL_DATA, "average_face_vert.npy"),
    vert_uv_path=os.path.join(PREDEFINED_MODEL_DATA, "vert_texcoords.npy"),
    output_dir="output_single_face"
):
    """
    Perform fitting for a single face.

    Args:
        detail_pts_path (str): Path to the detail_pts.npy file.
        ckpt_path (str): Path to the model checkpoint.
        average_face_path (str): Path to the average face file (default: predefined_model_data/average_face_vert.npy).
        vert_uv_path (str): Path to the vertex UV coordinates file (default: predefined_model_data/vert_texcoords.npy).
        output_dir (str): Directory to save the output files (default: output_single_face).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model and checkpoint
    config_dict = {
        "id_embedding_dropout_rate": 0.0,
        "exp_embedding_dropout_rate": 0.0,
        "uv_pixel_size": 128,
        "hash_encoding_n_levels": 16,
        "hash_encoding_n_features_per_level": 32,
        "hash_encoding_base_resolution": 4,
        "hash_encoding_finest_resolution": 128,
        "nonlinearity": "tanh_kx",
        "tanh_kx": 3,
        "surface_processor_deepsdf_like_network_mlp_out_dim_list": [512, 256, 256, 128, 32],
        "surface_processor_deepsdf_like_network_use_weight_norm": True,
        "surface_processor_deepsdf_like_network_dropout_rate": 0.0,
        "scale": True,
    }

    model = pixel_uv_embedding_ngp(
        args,
        id_num=280,
        exp_num=20,
        id_embedding_dim=128,
        exp_embedding_dim=128,
        id_embedding_dropout_rate=config_dict["id_embedding_dropout_rate"],
        exp_embedding_dropout_rate=config_dict["exp_embedding_dropout_rate"],
        uv_pixel_size=config_dict["uv_pixel_size"],
        hash_encoding_n_levels=config_dict["hash_encoding_n_levels"],
        hash_encoding_n_features_per_level=config_dict["hash_encoding_n_features_per_level"],
        hash_encoding_base_resolution=config_dict["hash_encoding_base_resolution"],
        hash_encoding_finest_resolution=config_dict["hash_encoding_finest_resolution"],
        nonlinearity=config_dict["nonlinearity"],
        tanh_kx=config_dict["tanh_kx"],
        surface_processor_deepsdf_like_network_mlp_out_dim_list=config_dict["surface_processor_deepsdf_like_network_mlp_out_dim_list"],
        surface_processor_deepsdf_like_network_use_weight_norm=config_dict["surface_processor_deepsdf_like_network_use_weight_norm"],
        surface_processor_deepsdf_like_network_dropout_rate=config_dict["surface_processor_deepsdf_like_network_dropout_rate"],
        scale=config_dict["scale"],
    ).cuda()
    model.basic_init_data_to_cuda()
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # Load detail points and average face
    detail_tu = torch.tensor(np.load(detail_pts_path)).reshape(12483, 3).cuda()
    average_face = torch.tensor(np.load(average_face_path)).cuda()
    delta_xyz = detail_tu - average_face
    vert_uv = torch.tensor(np.load(vert_uv_path)).reshape(12483, 2).cuda()

    # Perform fitting
    id_embedding_fitted, exp_embedding_fitted, fitted_delta_tu, mean_distance = model.fitting(vert_uv, delta_xyz)
    fitted_detail_tu = average_face + fitted_delta_tu

    # Save results
    model.mesh_save(fitted_detail_tu, os.path.join(output_dir, "fitted.obj"))
    model.mesh_save(detail_tu, os.path.join(output_dir, "gt.obj"))
    print(f"Fitting completed for {detail_pts_path}. Mean distance: {mean_distance:.5f}")


def main():
    single_face_fitting(
        detail_pts_path="", # Specify the path to your detail_pts.npy (test data) file
        ckpt_path="./experiment/best_encoder.pth.tar",
        output_dir="./output_single_face"
    )
    return
if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES']='1'
    # os.environ['CUDA_LAUNCH_BLOCKING']='1'
    main()


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


def create_dataloader(dataset_name, args, batch_size, shuffle=True):
    dataset = getattr(data_loader_for_pixel_form_with_raw_data, dataset_name)(args)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=8,
        pin_memory=True
    )

def load_model_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


def train(id_num, exp_num, ckpt_path=None):
    config_dict = {
        "id_embedding_dropout_rate": 0.0,
        "exp_embedding_dropout_rate": 0.0,
        "id_exp_embedding_loss_weight": 0.1,
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
        "uv_neighbor_smoothness_loss_weight": 0.0,
    }
    id_exp_embedding_loss_weight=config_dict["id_exp_embedding_loss_weight"]

    args.state = 'cm_3dmm'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    model = pixel_uv_embedding_ngp(
        args,
        # id/exp embedding set
        id_num=id_num,
        exp_num=exp_num,
        id_embedding_dim=128,
        exp_embedding_dim=128,
        id_embedding_dropout_rate=config_dict["id_embedding_dropout_rate"],
        exp_embedding_dropout_rate=config_dict["exp_embedding_dropout_rate"],
        # uv embedding set
        uv_pixel_size=config_dict["uv_pixel_size"],
        hash_encoding_n_levels=config_dict["hash_encoding_n_levels"],
        hash_encoding_n_features_per_level=config_dict["hash_encoding_n_features_per_level"],
        hash_encoding_base_resolution=config_dict["hash_encoding_base_resolution"],
        hash_encoding_finest_resolution=config_dict["hash_encoding_finest_resolution"],
        nonlinearity=config_dict["nonlinearity"],
        tanh_kx=config_dict["tanh_kx"],
        # implict function set
        surface_processor_deepsdf_like_network_mlp_out_dim_list=config_dict["surface_processor_deepsdf_like_network_mlp_out_dim_list"],
        surface_processor_deepsdf_like_network_use_weight_norm=config_dict["surface_processor_deepsdf_like_network_use_weight_norm"],
        surface_processor_deepsdf_like_network_dropout_rate=config_dict["surface_processor_deepsdf_like_network_dropout_rate"],
        scale=config_dict["scale"],
    ).cuda()

    model.basic_init_data_to_cuda()

    train_loader = create_dataloader(f"id_{id_num}_exp_{exp_num}_dataset_train", args, args.batch_size)
    test_loader = create_dataloader(f"id_{id_num}_exp_{exp_num}_dataset_test", args, args.batch_size)

    event_path = make_dirs(args)
    logger = configure_logging(event_path)

    args.pre_train=False
    args.num_epochs = 5000 if id_num * exp_num < 50 else 8000
    if args.pre_train:
        args.already_trained_epoch = 5882
        args.num_epochs = 8000
        model = load_model_checkpoint(model, ckpt_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    already_trained_epoch = args.already_trained_epoch;
    loss_train_per_epoch = [];
    loss_test_per_epoch = [];
    epoch_list = [];
    old_ckpt_path = ""
    newest_ckpt_path = ""
    best_loss = 10000;
    id_embedding_loss_weight = id_exp_embedding_loss_weight
    exp_embedding_loss_weight = id_exp_embedding_loss_weight

    for key, value in config_dict.items():
        logging.info(f"{key}:{value}")
    logging.info(f"ngp uv encoding")
    for epoch in tqdm(range(already_trained_epoch, args.num_epochs)):
        loss_train = [];
        loss_test = [];

        model.train();
        for i, input_dict in enumerate(train_loader):
            iter = epoch * len(train_loader) + i
            input_dict = dict_to_cuda(input_dict);
            predict_deform_dict = model(input_dict,)
            distance_loss = model.point_distance_loss(input_dict, predict_deform_dict)
            id_label_similarity_loss = model.id_label_similarity_loss_for_id_embeddings()
            id_embedding_mean, id_embedding_mean_loss = model.id_embedding_mean()
            id_embedding_std, id_embedding_std_loss = model.id_embedding_std()
            exp_label_similarity_loss = model.exp_label_similarity_loss_for_exp_embeddings()
            exp_embedding_std, exp_embedding_std_loss = model.exp_embedding_std()
            exp_embedding_mean, exp_embedding_mean_loss = model.exp_embedding_mean()
            loss_train.append(distance_loss.item());
            loss_total = distance_loss + \
                        id_embedding_loss_weight * (
                            id_label_similarity_loss + 0.3 * id_embedding_mean_loss + 0.3 * id_embedding_std_loss) + \
                        exp_embedding_loss_weight * (
                            exp_label_similarity_loss + 0.3 * exp_embedding_mean_loss + 0.3 * exp_embedding_std_loss)
            optimizer.zero_grad();
            loss_total.backward()
            optimizer.step()
            if i >= 0 and (i % args.writesummary == 0 or i < 5):
                logging.info(
                    (
                        f"train iter:{iter},lr rate:{scheduler.get_last_lr()[0]:.5f},"
                        f"distance loss:{distance_loss.item():.5f},"
                        f"id_loss:{id_embedding_loss_weight * (id_label_similarity_loss + 0.3 * id_embedding_mean_loss + 0.3 * id_embedding_std_loss).item():.5f},"
                        f"exp_loss:{exp_embedding_loss_weight * (exp_label_similarity_loss + 0.3 * exp_embedding_mean_loss + 0.3 * exp_embedding_std_loss).item():.5f},"
                        f"total loss:{loss_total.item():.5f}"
                    )
                )
        logging.info(f"total loss of epoch {epoch}: {torch.tensor(loss_train).mean():.5f}")
        scheduler.step()

        for j, input_dict in enumerate(test_loader):
            iter = epoch * len(test_loader) + j
            input_dict = dict_to_cuda(input_dict);
            predict_deform_dict = model(input_dict,)
            distance_loss = model.point_distance_loss(input_dict, predict_deform_dict)
            loss_test.append(distance_loss.item());
            if j >= 0 and (j % args.writesummary == 0 ):
                logging.info(
                    (
                        f"test iter:{iter},lr rate:{scheduler.get_last_lr()[0]:.6f},"
                        f"loss:{distance_loss.item():.5f}"
                    )
                )
        logging.info(f"test loss of epoch {epoch}: {torch.tensor(loss_test).mean():.5f}")


        if epoch == already_trained_epoch:
            best_loss_on_test_set = torch.tensor(loss_test).mean()
        if best_loss_on_test_set >= torch.tensor(loss_test).mean():
            save_state = {'state_dict': model.state_dict()}
            ckpt_path = os.path.join(event_path, 'checkpoint_path')
            best_loss_on_test_set = torch.tensor(loss_test).mean()
            logging.info("save best checkpoint on test set!")
            ckpt_path = os.path.join(event_path, 'checkpoint_path')
            torch.save(save_state, os.path.join(ckpt_path, 'best_' + 'encoder.pth.tar'), _use_new_zipfile_serialization=False)
            torch.save(save_state, os.path.join(ckpt_path, 'latest_' + str(epoch) + '_encoder.pth.tar'), _use_new_zipfile_serialization=False)
            if os.path.exists(old_ckpt_path):
                os.remove(old_ckpt_path)
            old_ckpt_path = os.path.join(ckpt_path, 'latest_' + str(epoch) + '_encoder.pth.tar')


        # draw
        loss_train_per_epoch.append(torch.tensor(loss_train).mean());
        loss_test_per_epoch.append(torch.tensor(loss_test).mean());
        epoch_list.append(epoch);
        for epoch_draw_iter in epoch_list:
            if loss_train_per_epoch[epoch_draw_iter - already_trained_epoch] > 10 * best_loss:
                loss_train_per_epoch[epoch_draw_iter - already_trained_epoch] = 0
            if loss_test_per_epoch[epoch_draw_iter - already_trained_epoch] > 10 * best_loss:
                loss_test_per_epoch[epoch_draw_iter - already_trained_epoch] = 0
        if epoch % 100 == 0:
            fig = plt.figure()
            pl.plot(epoch_list, loss_train_per_epoch, "r-", label="train_loss")
            pl.plot(epoch_list, loss_test_per_epoch, "g-", label="test_loss")
            pl.legend()
            pl.xlabel("epoch")
            pl.ylabel("loss")
            plt.savefig(os.path.join(ckpt_path, "loss.jpg"))
    return


def main():
    train(id_num=280,exp_num=20,)

    return
if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES']='1'
    # os.environ['CUDA_LAUNCH_BLOCKING']='1'
    main()


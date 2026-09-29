# -*- encoding: utf-8 -*-
"""
@File    :   test.py
@Time    :   2023/12/18 11:44:15
@Author  :   Xinyi Wu
@Version :   1.0
@Contact :   wuxinyi17@nudt.edu.cn
"""

"""Evaluation entry point; run with --mode test."""

import os

import torch

from data.dataloader import get_dataloader
from evaluation import Inference
from model.model import TextImgPersonReidNet
from option.options import options
from utils.read_write_data import load_checkpoint
from visualization import rank_token_visualization2


def main(opt):
    opt.device = torch.device("cuda:{}".format(opt.GPU_id))

    opt.save_path = "./checkpoints/{}/".format(opt.dataset) + opt.model_name

    gallery_loader, query_loader = get_dataloader(opt, "test")

    network = TextImgPersonReidNet(opt).to(opt.device)

    print("test_model:", opt.model_name)
    print("test dataset:", opt.dataset)

    checkpoint_feature_path = os.path.join(opt.save_path, "breakpoint/best_model")
    checkpoint_epoch_path = os.path.join(opt.save_path, "breakpoint/current_epoch")
    if os.path.exists(checkpoint_feature_path):
        model_state = load_checkpoint(checkpoint_feature_path)
        # print("model_state:", model_state)
        network.load_state_dict(model_state, strict=False)  # JIADJIA
        checkpoint_epoch = torch.load(checkpoint_epoch_path)
        print("Testing: Best Model Checkpoint successfully loaded!")
        print("Best Model Epoch:{}".format(checkpoint_epoch))
        # model test
        network.eval()
        # model_test
        cmc, mAP, mINP, cmc2, mAP2, mINP2, all_infer_time, infer_time_per_query = Inference(
            opt, 1, network, gallery_loader, query_loader
        )
        str1 = "Testing: i2t: @R1: {:.4}, @R5: {:.4}, @R10: {:.4}, mAP: {:.4}, mINP: {:.4}".format(
            cmc[0], cmc[4], cmc[9], mAP, mINP
        )
        str2 = "Testing: t2i: @R1: {:.4}, @R5: {:.4}, @R10: {:.4}, mAP: {:.4}, mINP: {:.4}".format(
            cmc2[0], cmc2[4], cmc2[9], mAP2, mINP2
        )

        print(str1)
        print(str2)
        # rank list visualize & token visualize

        # rank_token_visualization(opt, network, gallery_loader, query_loader)
        rank_token_visualization2(opt, network, gallery_loader, query_loader)

        torch.cuda.empty_cache()
    else:
        print("no saved model!")


if __name__ == "__main__":
    main(options().opt)

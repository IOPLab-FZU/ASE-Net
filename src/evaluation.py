# -*- encoding: utf-8 -*-
"""
@File    :   test.py
@Time    :   2023/12/18 11:44:15
@Author  :   Xinyi Wu
@Version :   1.0
@Contact :   wuxinyi17@nudt.edu.cn
"""

"""Feature extraction and bidirectional retrieval evaluation."""

import re
import time

import torch

from metrics.metric import R1_mAP_eval


def pre_caption(caption_list):
    caption_list2 = []
    for caption in caption_list:
        caption = (
            re.sub(
                r"([,.'!?\"()*#:;~])",
                " ",
                caption.lower(),
            )
            .replace("-", " ")
            .replace("/", " ")
            .replace("<person>", "person")
        )

        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        caption_list2.append(caption)

    return caption_list2


def feature_gene(opt, network, gallery_loader, query_loader):
    image_feature_global_list = []
    patch_feature_part_list = []
    image_selected_indices_list = []
    img_labels = []

    for times, [image, label] in enumerate(gallery_loader):
        image = image.to(opt.device)
        label = label.to(opt.device)

        with torch.no_grad():
            if opt.mode == "train":
                img_global_i, patch_part_i = network.img_embedding(image)
                selected_indices = torch.tensor([])
            elif opt.mode == "test":
                img_global_i, patch_part_i, selected_indices = network.img_embedding(image)
                # [bs, Rv*49]
        image_feature_global_list.append(img_global_i)
        patch_feature_part_list.append(patch_part_i)
        image_selected_indices_list.append(selected_indices)
        img_labels.append(label.view(-1))

    img_labels = torch.cat(img_labels, 0)
    image_selected_indices = torch.cat(image_selected_indices_list, 0)

    text_feature_global_list = []
    word_feature_part_list = []
    txt_labels = []
    word_tokens_list = []
    select_tokens_list = []
    attention_weight_list = []
    for times, [text, label] in enumerate(query_loader):
        # text = text.to(opt.device)
        label = label.to(opt.device)

        with torch.no_grad():
            text = list(text)
            text = pre_caption(text)
            if opt.mode == "train":
                text_global_i, word_part_i = network.txt_embedding(text)
                attention_cls_part = torch.tensor([])
                word_tokens = []
                select_token_indices = []
            elif opt.mode == "test":  # TODO
                (
                    text_global_i,
                    word_part_i,
                    attention_cls_part,
                    word_tokens,
                    select_token_indices,
                ) = network.txt_embedding(text)

        text_feature_global_list.append(text_global_i)
        word_feature_part_list.append(word_part_i)
        attention_weight_list.append(attention_cls_part)
        txt_labels.append(label.view(-1))
        word_tokens_list += word_tokens
        select_tokens_list += select_token_indices

    txt_labels = torch.cat(txt_labels, 0)
    attention_weights = torch.cat(attention_weight_list, 0)

    img_labels = img_labels.cpu()
    txt_labels = txt_labels.cpu()

    return (
        text_feature_global_list,
        word_feature_part_list,
        image_feature_global_list,
        patch_feature_part_list,
        txt_labels,
        img_labels,
        image_selected_indices,
        attention_weights,
        word_tokens_list,
        select_tokens_list,
    )


def Inference(opt, epoch, network, gallery_loader, query_loader):
    evaluator = R1_mAP_eval(max_rank=20)
    infer_start_t = time.time()
    (
        text_feature_global_list,
        word_feature_part_list,
        image_feature_global_list,
        patch_feature_part_list,
        txt_labels,
        img_labels,
        image_selected_indices,
        attention_weights,
        word_tokens_list,
        select_tokens_list,
    ) = feature_gene(opt, network, gallery_loader, query_loader)
    query_length = len(txt_labels)
    cmc, mAP, mINP = evaluator.compute(
        network,
        text_feature_global_list,
        word_feature_part_list,
        image_feature_global_list,
        patch_feature_part_list,
        txt_labels,
        img_labels,
    )
    cmc2, mAP2, mINP2 = evaluator.compute2(
        network,
        text_feature_global_list,
        word_feature_part_list,
        image_feature_global_list,
        patch_feature_part_list,
        txt_labels,
        img_labels,
    )
    all_infer_time = time.time() - infer_start_t
    infer_time_per_query = all_infer_time / query_length

    return cmc, mAP, mINP, cmc2, mAP2, mINP2, all_infer_time, infer_time_per_query

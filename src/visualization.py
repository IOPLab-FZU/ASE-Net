# -*- encoding: utf-8 -*-
"""
@File    :   test.py
@Time    :   2023/12/18 11:44:15
@Author  :   Xinyi Wu
@Version :   1.0
@Contact :   wuxinyi17@nudt.edu.cn
"""

"""Retrieval rankings and image/text attention plots."""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from torchvision import transforms

from evaluation import feature_gene
from metrics.metric import R1_mAP_eval
from utils.data_paths import change_path
from utils.read_write_data import read_json, write_txt

t_resize = transforms.Compose([transforms.Resize((224, 224), Image.BICUBIC)])


def Image_Token_Selection1(background_img_path, attention_mask_indices, save_path):
    image = Image.open(background_img_path)
    image = t_resize(image)
    image = np.asarray(image)
    # VIT-32
    image_tokens = image.reshape(7, 32, 7, 32, 3).swapaxes(1, 2).reshape(49, 32, 32, 3)
    for i in range(49):
        if i not in attention_mask_indices:
            image_tokens[i] = 0.2 * image_tokens[i] + 0.8 * 255

    image_t = image_tokens.reshape(7, 7, 32, 32, 3).swapaxes(1, 2).reshape(224, 224, 3)
    img_show = np.concatenate([image] + [image_t], axis=1)
    plt.figure(figsize=(10, 5))
    plt.imshow(img_show)
    plt.savefig(save_path)
    plt.axis("off")

    return 0


def Image_Token_Selection(background_img_path, attention_mask_indices, save_path):
    # 1. 加载并预处理图像
    image = Image.open(background_img_path).convert("RGB")
    image = t_resize(image)
    image_np = np.asarray(image)

    # 2. 创建平滑的注意力权重图
    # 初始化注意力权重矩阵 (7x7)
    attention_weights = np.zeros((7, 7))
    for idx in attention_mask_indices:
        row = idx // 7
        col = idx % 7
        attention_weights[row, col] = 1.0  # 基础权重值

    # 3. 应用高斯滤波实现平滑效果
    # 调整 sigma 参数控制平滑程度，值越大越平滑
    smooth_weights = gaussian_filter(attention_weights, sigma=1.0)

    # 4. 将 7x7 权重图插值到 224x224 尺寸
    attention_map = zoom(smooth_weights, 32, order=3)  # 三次样条插值保证平滑

    # 5. 归一化权重图到 [0,1] 范围
    norm_attention = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min() + 1e-8
    )

    # 6. 应用热力图色彩映射 (使用 jet 色彩方案，也可尝试 viridis 等)
    cmap = plt.get_cmap("jet")
    heatmap = cmap(norm_attention)[:, :, :3] * 255  # 转换为 RGB

    # 7. 将热力图与原始图像融合
    # 调整 alpha 参数控制热力图的透明度
    alpha = 0.6
    blended = (image_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)

    # 8. 创建包含原始图像和注意力图的拼接图
    img_show = np.concatenate([image_np, blended], axis=1)

    # 9. 绘制图像并添加颜色条说明注意力强度
    plt.figure(figsize=(12, 6))
    plt.imshow(img_show)
    plt.axis("off")

    # 添加颜色条
    cbar_ax = plt.axes([0.25, 0.05, 0.5, 0.03])  # 位置和大小
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Attention Intensity")

    # 10. 保存图像
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    return 0


def Text_Token_Visualize(word_labels, attention_weights, select_token_indices, save_path):
    # 绘制热力图
    fig, ax = plt.subplots()

    im = ax.imshow(np.array([attention_weights]), cmap="YlGnBu")  # YlGnBu

    # 添加标签
    fig.set_figheight(3)
    fig.set_figwidth(10)
    ax.tick_params(axis="x", labelsize=14)  # 调整x轴标签字体大小
    ax.set_xticks(np.arange(len(attention_weights)))
    ax.set_yticks([0])
    ax.set_xticklabels(word_labels, rotation=90, fontsize=12)

    # 添加边界检查，防止索引越界
    # xticklabels = ax.get_xticklabels()
    # for w in select_token_indices:
    # if 0 <= w < len(xticklabels):
    # xticklabels[w].set_bbox(dict(facecolor='red', edgecolor='red', alpha=0.3, linewidth=1))
    ax.set_yticklabels([""])
    ax.set_title("Attention Weights of Each Word")

    # 添加热力分布bar
    # cbar = ax.figure.colorbar(im, ax=ax)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7)
    cbar.ax.set_ylabel("Heat distribution", rotation=-90, va="bottom")
    # 保存图像
    plt.savefig(save_path)
    plt.close()  # 关闭图形以释放内存


def rank_token_visualization(opt, network, gallery_loader, query_loader):
    evaluator = R1_mAP_eval(max_rank=20)
    (
        text_feature_global_list,
        word_feature_part_list,
        image_feature_global_list,
        patch_feature_part_list,
        txt_labels,
        img_labels,
        image_selected_indices,
        word_attention_weights,
        word_tokens_list,
        select_tokens_list,
    ) = feature_gene(opt, network, gallery_loader, query_loader)

    # visualize data generation
    data_save = read_json(os.path.join(opt.dataroot, "processed_data/test_save.json"))
    img_path_list = [
        change_path(opt.dataset, os.path.join(opt.dataroot, img_path))
        for img_path in data_save["img_path"]
    ]
    caption_list = data_save["captions"]

    ##Step1: Rank_visualization
    distmat, q_pids, g_pids = evaluator.similarity_compute(
        network,
        text_feature_global_list,
        word_feature_part_list,
        image_feature_global_list,
        patch_feature_part_list,
        txt_labels,
        img_labels,
    )

    num_q, num_g = distmat.shape
    gallery_indices = np.argsort(-distmat, axis=1)  # [num_q, num_g]  返回相似度从大到小排序后的索引
    gallery_indices_top10 = gallery_indices[:, :10]
    visualization_log = os.path.join(opt.save_path, opt.dataset + "_visualize_log.txt")

    for q_idx in range(num_q):  # 遍历每个query
        write_txt(
            f"query index: {q_idx}, caption_label: {q_pids[q_idx]}, caption: {caption_list[q_idx]} \n ",
            visualization_log,
        )
        g_idx_top10 = gallery_indices_top10[q_idx]  # (10, )

        for g_idx in range(10):
            gg = g_idx_top10[g_idx]
            write_txt(
                f"matched gallery index: {gg}, img_label:{g_pids[gg]}, gallery_img_path: {img_path_list[gg]} \n",
                visualization_log,
            )
    print("visualization_log successfully generated!")

    if opt.Topk_Selection:
        ##Step 2: Image Token Selection Visualization
        for i in range(len(img_path_list)):
            print(i)
            print(img_path_list[i])
            # image_selected_indices     #被选中的patch索引
            ww = img_path_list[i].split("/")
            # print(ww)
            img_name = ww[-1]
            if "bmp" in img_name:
                img_name = img_name.replace("bmp", "jpg")
            save_path = os.path.join(opt.save_path, "image_visualization/")
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + img_name
            print("save_path:", save_path)
            Image_Token_Selection(img_path_list[i], image_selected_indices[i], save_path)

        # Step 3: Word Token Selection Visualization

        word_attention_weights = word_attention_weights.detach().cpu()  # [num_texts, 25]

        text_name_list = []
        for q in range(len(img_path_list)):
            ww = img_path_list[q].split("/")[-1]
            text_name = ww.split(".")[0]
            text_name_list.append(text_name + "_1.png")
            text_name_list.append(text_name + "_2.png")

        for j in range(len(word_tokens_list)):
            # 检查索引是否在范围内
            if j >= len(text_name_list):
                print(
                    f"Warning: text_name_list index {j} out of range (length: {len(text_name_list)})"
                )
                break

            word_labels = word_tokens_list[j]
            word_labels2 = []
            for word_labels_i in word_labels:
                word_labels_i = word_labels_i[:-4]  # 去掉</w>
                word_labels2.append(word_labels_i)

            if len(word_labels2) > opt.max_words:
                word_labels2 = word_labels2[: opt.max_words]
            elif len(word_labels2) < opt.max_words:
                yhi = opt.max_words - len(word_labels2)
                for w in range(yhi):
                    word_labels2.append("<pad>")

            # print("word_labels:", word_labels2)
            attention_weights = word_attention_weights[j].tolist()  # list  len = 25
            # print("attention_weights:",attention_weights)
            select_token_indices = select_tokens_list[j]  # len = 25*Rt
            save_path = os.path.join(opt.save_path, "text_visualization/")
            os.makedirs(save_path, exist_ok=True)

            save_path = save_path + text_name_list[j]

            # Text_Token_Visualize(word_labels2, attention_weights, select_token_indices, save_path)

    return 0


def rank_token_visualization2(opt, network, gallery_loader, query_loader):
    evaluator = R1_mAP_eval(max_rank=20)
    (
        text_feature_global_list,
        word_feature_part_list,
        image_feature_global_list,
        patch_feature_part_list,
        txt_labels,
        img_labels,
        image_selected_indices,
        word_attention_weights,
        word_tokens_list,
        select_tokens_list,
    ) = feature_gene(opt, network, gallery_loader, query_loader)

    # visualize data generation
    data_save = read_json(os.path.join(opt.dataroot, "processed_data/test_save.json"))
    img_path_list = [
        change_path(opt.dataset, os.path.join(opt.dataroot, img_path))
        for img_path in data_save["img_path"]
    ]
    caption_list = data_save["captions"]

    ##Step1: Rank_visualization
    distmat, q_pids, g_pids = evaluator.similarity_compute2(
        network,
        text_feature_global_list,
        word_feature_part_list,
        image_feature_global_list,
        patch_feature_part_list,
        txt_labels,
        img_labels,
    )

    num_q, num_g = distmat.shape
    query_indices = np.argsort(-distmat, axis=1)  # [num_q, num_g]  返回相似度从大到小排序后的索引
    query_indices_top10 = query_indices[:, :10]
    visualization_log = os.path.join(opt.save_path, opt.dataset + "_visualize_log2.txt")

    for g_idx in range(num_g):  # 遍历每个图像查询 (gallery as query)
        write_txt(
            f"gallery query index: {g_idx}, img_label: {g_pids[g_idx]}, gallery_img_path: {img_path_list[g_idx]} \n",
            visualization_log,
        )
        q_idx_top10 = query_indices_top10[g_idx]  # (10, ) 假设这是与图像查询最匹配的10个文本索引

        for rank, q_idx in enumerate(q_idx_top10):  # 遍历前10个匹配的文本
            write_txt(
                f"matched query index: {q_idx}, query_label: {q_pids[q_idx]}, query_text_path: {caption_list[q_idx]} \n",
                visualization_log,
            )
            # 可选：如果您想记录排名信息，可以添加：
            write_txt(f"Rank {rank + 1}: ", visualization_log)  # 排名从1开始更符合习惯

    print("visualization_log successfully generated!")

    return 0

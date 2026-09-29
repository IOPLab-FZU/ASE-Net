"""Dataset-specific occlusion paths shared by loading and visualization."""


def change_path(dataset_name, pathss):
    if dataset_name == "ICFG-PEDES":
        if "train" in pathss:
            occlusion_path = pathss.replace("train", "train_occlusion_new")
        elif "test" in pathss:
            occlusion_path = pathss.replace("test", "test_occlusion_new")
        else:
            occlusion_path = pathss
    elif dataset_name == "CUHK-PEDES":
        if "cam_a" in pathss:
            occlusion_path = pathss.replace("cam_a", "cam_a_occlusion_new")
        elif "cam_b" in pathss:
            occlusion_path = pathss.replace("cam_b", "cam_b_occlusion_new")
        elif "CUHK01" in pathss:
            occlusion_path = pathss.replace("CUHK01", "CUHK01_occlusion_new")
        elif "CUHK03" in pathss:
            occlusion_path = pathss.replace("CUHK03", "CUHK03_occlusion_new")
        elif "Market" in pathss:
            occlusion_path = pathss.replace("Market", "Market_occlusion_new")
        elif "test_query" in pathss:
            occlusion_path = pathss.replace("test_query", "test_query_occlusion_new")
        elif "train_query" in pathss:
            occlusion_path = pathss.replace("train_query", "train_query_occlusion_new")
        else:
            occlusion_path = pathss
    elif dataset_name == "RSITMD":
        if "imgs" in pathss:
            occlusion_path = pathss.replace("imgs", "imgs")
        else:
            occlusion_path = pathss
    return occlusion_path

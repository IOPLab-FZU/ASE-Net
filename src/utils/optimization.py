"""Optimizer parameter groups and backbone learning-rate multipliers."""

from torch import optim


def build_optimizer(opt, network, id_loss_fun):
    """Keep the original backbone, head and ID-loss learning rates."""
    ft_params = []
    param_groups = []

    if opt.image_fintune:
        if opt.img_model == "ResNet50":
            ft_params += list(map(id, network.ImageExtract.parameters()))
        else:
            ft_params += list(map(id, network.ImageExtract.cnn.parameters()))
    if opt.text_fintune:
        if opt.txt_model == "ResNet50":
            ft_params += list(map(id, network.TextExtract.parameters()))
        else:
            ft_params += list(map(id, network.TextExtract.textExtractor.parameters()))

    other_params = list(
        filter(lambda p: id(p) not in ft_params, network.parameters())
    )  # backbone fc
    if opt.ID_LOSS:
        id_loss_params = list(id_loss_fun.parameters())

    param_groups.append({"params": other_params, "lr": opt.lr})  # 10-4 0.0001
    if opt.ID_LOSS:
        param_groups.append({"params": id_loss_params, "lr": 5 * opt.lr})  # 5*10-4

    if opt.image_fintune:
        if opt.img_model == "ResNet50":
            param_groups.append(
                {"params": network.ImageExtract.parameters(), "lr": opt.lr * 0.1}
            )  # 10-5
        else:
            param_groups.append(
                {"params": network.ImageExtract.cnn.parameters(), "lr": opt.lr * 0.1}
            )  # 10-5
    if opt.text_fintune:
        if opt.txt_model == "ResNet50":
            param_groups.append({"params": network.TextExtract.parameters(), "lr": opt.lr * 0.1})
        else:
            param_groups.append(
                {
                    "params": network.TextExtract.textExtractor.parameters(),
                    "lr": opt.lr * 0.1,
                }
            )

    optimizer = optim.Adam(param_groups, betas=(opt.adam_alpha, opt.adam_beta))

    return optimizer

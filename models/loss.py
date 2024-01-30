import torch.nn.functional as F
import torch


def model_loss_train(disp_ests, disp_gts, img_masks):
    weights = [1.0, 0.3, 1.0]
    all_losses = []
    for disp_est, disp_gt, weight, mask_img in zip(disp_ests, disp_gts, weights, img_masks):
        # print("disp_est", disp_est.shape)
        # print("disp_gt", disp_gt.shape)
        # print("mask_img", mask_img.shape)
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask_img], disp_gt[mask_img], size_average=True))
    return sum(all_losses)


def model_loss_test(disp_ests, disp_gts, img_masks):
    weights = [1.0]
    all_losses = []
    for disp_est, disp_gt, weight, mask_img in zip(disp_ests, disp_gts, weights, img_masks):
        all_losses.append(weight * F.l1_loss(disp_est[mask_img], disp_gt[mask_img], size_average=True))
    return sum(all_losses)
import torch, cv2, os, sys, numpy as np, matplotlib.pyplot as plt
from PIL import Image

from ModelZoo.utils import load_as_tensor, Tensor2PIL, PIL2Tensor, _add_batch_one
from ModelZoo import NN_LIST, get_model, load_model, print_network
from SaliencyModel.utils import vis_saliency, vis_saliency_kde, click_select_position, grad_abs_norm, grad_norm, prepare_images, make_pil_grid, blend_input
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from SaliencyModel.attributes import attr_grad
from SaliencyModel.BackProp import I_gradient, attribution_objective, Path_gradient
from SaliencyModel.BackProp import saliency_map_PG as saliency_map
from SaliencyModel.BackProp import GaussianBlurPath
from SaliencyModel.utils import grad_norm, IG_baseline, interpolation, isotropic_gaussian_kernel

model_name = 'SwinIR'
window_size = 16 # 80  # Define windoes_size of D
image_path = './test_images/7.png'
w = 175  # The x coordinate of your select patch, 125 as an example
h = 130  # The y coordinate of your select patch, 160 as an example
        # And check the red box
        # Is your selected patch this one? If not, adjust the `w` and `h`.
image_name = os.path.basename(image_path)[:-4]
idx = 2

for model_variants in ['SRx4_win8', 'SRx4_win16', 'SRx4_win32']:
    model = load_model(model_name + '@' + model_variants)

    img_lr, img_hr = prepare_images(image_path)  # Change this image name
    tensor_lr = PIL2Tensor(img_lr)[:3] ; tensor_hr = PIL2Tensor(img_hr)[:3]
    cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2) ; cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2)

    draw_img = pil_to_cv2(img_hr)
    cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
    position_pil = cv2_to_pil(draw_img)

    sigma = 1.2 ; fold = 50 ; l = 9 ; alpha = 0.5
    attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), model, attr_objective, gaus_blur_path_func, cuda=True)
    grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
    abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=4)
    saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
    blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    pil = make_pil_grid(
        [position_pil,
        saliency_image_abs,
        blend_abs_and_input,
        blend_kde_and_input,
        Tensor2PIL(torch.clamp(torch.from_numpy(result), min=0., max=1.))]
    )

    pil.save('./Results/' + '_'.join([model_name, model_variants, image_name, str(idx)]) + '.png')

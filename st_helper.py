from glob import glob

import shutil

import torch
import torch.nn.functional as F
import torch.optim as optim
from imageio import imwrite
from torch.autograd import Variable

import shutil
import utils
from pyr_lap import *
from stylize_objectives import objective_class
from utils import *
from vgg_pt import *


def style_transfer(stylized_im, content_im, style_path, output_path, scl, long_side, mask, content_weight=0.,
                   use_guidance=False, regions=0, coords=0, lr=2e-3, palette_content=False,
                   lower_layers_only=False):

    REPORT_INTERVAL = 100
    RESAMPLE_FREQ = 1
    MAX_ITER = 250
    save_ind = 0

    temp_name = './'+output_path.split('/')[-1].split('.')[0]+'_temp.png'

    # Keep track of current output image for GUI
    canvas = aug_canvas(stylized_im, scl, 0)
    imwrite(temp_name, canvas)
    shutil.move(temp_name, output_path)

    # Define feature extractor
    cnn = utils.to_device(Vgg16_pt())
    phi = lambda x: cnn.forward(x)
    phi2 = lambda x, y, z: cnn.forward_cat(x, z, samps=y, forward_func=cnn.forward)
    phi_lower_layers_only = lambda x: cnn.forward(x, lower_layers_only=True)

    # Define Optimizer (Optimize over laplacian pyramid instead of pixels directly)
    s_pyr = dec_lap_pyr(stylized_im, 5)
    s_pyr = [Variable(li.data, requires_grad=True) for li in s_pyr]
    optimizer = optim.RMSprop(s_pyr, lr=lr)

    # Pre-Extract Content Features
    z_c = phi(content_im) if not lower_layers_only else phi_lower_layers_only(content_im)
    print(f"Extract content features. Using lower layers only: {lower_layers_only}. Len of z_c: {len(z_c)}")

    # Pre-Extract Style Features from a Folder
    paths = glob(style_path+'*')[::3]

    # Create Objective Object
    objective_wrapper = objective_class()

    # Extract style features
    z_s_all = []
    for ri in range(len(regions[1])):  # = number of style images = 1
        z_s, style_ims = load_style_folder(phi2, paths, regions, ri, n_samps=-1, subsamps=1000, scale=long_side,
                                           inner=5)
        z_s_all.append(z_s)

    # Extract guidance features if required
    gs = np.array([0.])
    if use_guidance:
        gs = load_style_guidance(phi, style_path, coords[:,2:], scale=long_side)

    # Randomly choose spatial locations to extract features from
    # Reconstruct image from pyramid by successive bilinear upsampling (use pyramid)
    stylized_im = syn_lap_pyr(s_pyr)

    for ri in range(len(regions[0])):  # = number of content images?
        # Initializes the random positions in objective_wrapper

        r_temp = regions[0][ri]
        r_temp = torch.from_numpy(r_temp).unsqueeze(0).unsqueeze(0).contiguous()
        r = F.upsample(r_temp, (stylized_im.size(3), stylized_im.size(2)), mode='bilinear')[0, 0, :, :].numpy()

        if r.max() < 0.1:
            r = np.greater(r+1., 0.5)
        else:
            r = np.greater(r, 0.5)

        objective_wrapper.init_inds(z_c, z_s_all, r, ri)

    if use_guidance:
        objective_wrapper.init_g_inds(coords, stylized_im)

    for i in range(MAX_ITER):

        # zero out gradients and compute output image from pyramid
        optimizer.zero_grad()
        stylized_im = syn_lap_pyr(s_pyr)  # use pyramid

        # Dramatically re-sample Large Set of Spatial Locations
        if i == 0 or i % (RESAMPLE_FREQ*10) == 0:
            for ri in range(len(regions[0])):
                
                r_temp = regions[0][ri]
                r_temp = torch.from_numpy(r_temp).unsqueeze(0).unsqueeze(0).contiguous()
                r = F.upsample(r_temp, (stylized_im.size(3), stylized_im.size(2)), mode='bilinear')[0, 0, :, :].numpy()

                if r.max() < 0.1:
                    r = np.greater(r+1., 0.5)
                else:
                    r = np.greater(r, 0.5)

                objective_wrapper.init_inds(z_c, z_s_all, r, ri)

        # Subsample spatial locations to compute loss over
        if i == 0 or i % RESAMPLE_FREQ == 0:
            objective_wrapper.shuffle_feature_inds()
        
        # Extract Features from Current Output
        z_x = phi(stylized_im)
        z_x_lower_layers_only = phi_lower_layers_only(stylized_im) if lower_layers_only else None

        # Compute Objective and take gradient step
        ell = objective_wrapper.eval(z_x, z_x_lower_layers_only, z_c, z_s_all, gs, content_weight=content_weight,
                                     moment_weight=1.0,
                                     palette_content=palette_content)

        ell.backward()
        optimizer.step()
            
        # Periodically save output image for GUI ###
        if (i+1) % 10 == 0:
            canvas = aug_canvas(stylized_im, scl, i)
            imwrite(temp_name, canvas)
            shutil.move(temp_name, output_path)

        # Periodically Report Loss and Save Current Image
        if (i+1) % REPORT_INTERVAL == 0:
            print((i+1), ell)
            save_ind += 1

    return stylized_im, ell

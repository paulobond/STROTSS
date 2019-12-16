import torch
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import time

from st_helper import *
from utils import *


def run_st(content_path, style_path, content_weight, max_scl, coords, use_guidance, regions,
           output_path='./output.png', palette_content=False, lower_layers_only=False):

    smll_sz = 64
    start = time.time()

    for scl in range(1, max_scl):

        long_side = smll_sz*(2**(scl-1))
        lr = 2e-3

        # Load Style and Content Image
        content_im = utils.to_device(
            Variable(load_path_for_pytorch(content_path, long_side, force_scale=True).unsqueeze(0)))
        content_im_mean = utils.to_device(
            Variable(load_path_for_pytorch(style_path,long_side, force_scale=True).unsqueeze(0)))\
            .mean(2, keepdim=True).mean(3, keepdim=True)
        
        # Compute bottom level of Laplacian pyramid for content image at current scale
        lap = content_im.clone() -\
              F.upsample(F.upsample(content_im, (content_im.size(2)//2, content_im.size(3)//2), mode='bilinear'),
                         (content_im.size(2), content_im.size(3)), mode='bilinear')

        # Initialize by zeroing out all but highest and lowest levels of Laplacian Pyramid #
        if scl == 1:
            stylized_im = Variable(content_im_mean+lap)
        # Otherwise bilinearly upsample previous scales output and add back bottom level of Laplacian
        # pyramid for current scale of content image
        if 1 < scl < max_scl-1:
            stylized_im = F.upsample(stylized_im.clone(), (content_im.size(2), content_im.size(3)), mode='bilinear')+lap
        if scl > 3:
            stylized_im = F.upsample(stylized_im.clone(), (content_im.size(2), content_im.size(3)), mode='bilinear')
            lr = 1e-3

        # Style Transfer at this scale
        stylized_im, final_loss = style_transfer(stylized_im, content_im, style_path, output_path, scl, long_side, 0.,
                                                 use_guidance=use_guidance, coords=coords,
                                                 content_weight=content_weight, lr=lr, regions=regions,
                                                 palette_content=palette_content, lower_layers_only=lower_layers_only)

        # Decrease Content Weight for next scale (alpha)
        content_weight = content_weight/2.0

    print("Finished in: ", int(time.time()-start), 'Seconds')
    print('Final Loss:', final_loss)

    canvas = torch.clamp(stylized_im[0], -0.5, 0.5).data.cpu().numpy().transpose(1, 2, 0)
    imwrite(output_path, canvas)
    return final_loss, stylized_im


if __name__=='__main__':

    # Parse Command Line Arguments
    content_path = sys.argv[1]
    style_path = sys.argv[2]
    content_weight = float(sys.argv[3])*16.0
    max_scl = 5

    use_guidance_region = '-gr' in sys.argv
    use_guidance_points = False
    use_gpu = not ('-cpu' in sys.argv) and torch.cuda.is_available()
    print("Use GPU:  " + str(use_gpu))
    utils.use_gpu = use_gpu

    paths = glob(style_path+'*')
    losses = []
    ims = []

    palette_content = '-orgclr' in sys.argv
    content_loss_lower_layers_only = '-ctllo' in sys.argv

    if content_loss_lower_layers_only:
        print("******** EXPERIMENT: USING LOWER LAYERS ONLY TO COMPUTE CONTENT LOSS **********")
    else:
        print("******** USING ALL LAYERS TO COMPUTE CONTENT LOSS AND STYLE LOSS **********")

    if '-output' in sys.argv:
        output_path = sys.argv[sys.argv.index('-output') + 1]
    else:
        output_path = './output.png'

    './output.png'

    # Preprocess User Guidance if Required
    coords=0.
    if use_guidance_region:
        i = sys.argv.index('-gr')
        regions = utils.extract_regions(sys.argv[i+1], sys.argv[i+2])
    else:
        try:
            regions = [[imread(content_path)[:, :, 0]*0.+1.], [imread(style_path)[:, :, 0]*0.+1.]]
        except Exception:
            regions = [[imread(content_path)[:, :]*0.+1.], [imread(style_path)[:, :]*0.+1.]]

    # Style Transfer and save output
    loss, canvas = run_st(content_path, style_path, content_weight, max_scl, coords, use_guidance_points, regions,
                          palette_content=palette_content, lower_layers_only=content_loss_lower_layers_only,
                          output_path=output_path)

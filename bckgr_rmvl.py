import os
from skimage import io, transform
import requests
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import glob

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET

# Normalize tensor (torch version)
def normalize(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name, mas, d_dir):
    # normalize (-#,+#) -> (0.0,1.0)
    mask = mas
    mask = mask.squeeze()
    # sacle (0.0, 1.0) -> (0, 255)
    mask_np = mask.cpu().data.numpy()

    mask = Image.fromarray(mask_np*255).convert("L")
    img_name = os.path.split(image_name)[-1]
    image = Image.open(image_name).convert('RGBA')
    
    mask = mask.resize(image.size, resample=Image.BILINEAR)

    image.putalpha(mask)
    
    imidx, _ = os.path.splitext(img_name)
    image.save(os.path.join(d_dir, imidx+'.png'))


def main():
    # ----------- imgs path and name --------------
    model_name='u2net'

    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_human_imgs')
    mask_dir = os.path.join(os.getcwd(), 'test_data', 'test_human_imgs_results')
    model_dir = os.path.join(os.getcwd(), 'weights', model_name+'_bckgr_rmvl', model_name + '_bckgr_rmvl.pth')
    img_name_list = glob.glob(os.path.join(image_dir, '*'))
    print(img_name_list)

    # ----------- data_loader -------------------
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
            lbl_name_list = [],
            transform=transforms.Compose([RescaleT(320),
                ToTensorLab(flag=0)])
            )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1)

    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    net.eval()
    
    # ------------ inference for each img ---------------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print('inferencing:', img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        # feed to model
        d1, _, _, _, _, _, _ = net(inputs_test)

        # recieve d1 mask
        mas = d1[:,0,:,:]
        mas = normalize(mas)
    
        # save result
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir, exist_ok=True)
        save_output(img_name_list[i_test], mas, mask_dir)

        del d1

if __name__ == "__main__":
    main()


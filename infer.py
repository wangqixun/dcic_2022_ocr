import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
import torchvision.transforms as transforms
from transformers import AutoModel
from transformers import AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler

from captcha.image import ImageCaptcha
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict
from rich import print
import os
from PIL import Image
import time
from ctc_decoder import best_path, beam_search
from ctcdecode import CTCBeamDecoder

import string

from model.convnext import convnext_base
from main import Model, decode

width, height = 320, 128
n_input_length = 80
characters = '-' + string.digits + string.ascii_letters
n_len, n_classes = 4, len(characters)



class CaptchaDataset(Dataset):
    def __init__(self, characters, width, height, input_length, label_length, mode='A'):
        super(CaptchaDataset, self).__init__()
        self.characters = characters
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)
        self.mode = mode
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if self.mode == 'A':
            self.val_dir = '/new_share/wangqixun/workspace/bs/dcic_ocr/data/test_dataset/'
            self.val_names = [f"{i+1}.png" for i in range(len(os.listdir(self.val_dir)))]
            self.transform_val = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])

    def __len__(self):
        return len(self.val_names)

    def __getitem__(self, index):
        if self.mode == 'A':
            img_file = os.path.join(self.val_dir, self.val_names[index])
            image = self.transform_val(Image.open(img_file).convert('RGB'))
            target = ''
            input_length = torch.full(size=(1, ), fill_value=self.input_length, dtype=torch.long)
            target_length = torch.full(size=(1, ), fill_value=self.label_length, dtype=torch.long)
            return image, target, input_length, target_length


def ctc_decode(ctc_input):
    ctc_input = ctc_input.cpu().numpy()
    res = []
    for idx in range(len(ctc_input)):
        r = beam_search(ctc_input[idx], characters[1:], beam_width=3)
        res.append(r)
    return res


def valid(model, dataloader):
    submit_file = '/new_share/wangqixun/workspace/bs/dcic_ocr/submit/1.csv'
    model_1, model_2, model_3 = model
    model_1.eval()
    model_2.eval()
    model_3.eval()

    decoder = CTCBeamDecoder(
        characters,
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=20,
        num_processes=4,
        blank_id=0,
        log_probs_input=False
    )


    print(f"=> test")
    res = []
    with torch.no_grad():
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(dataloader):
            # if batch_index>10:
            #     break
            data = data.cuda()

            output = model_1(data)
            output_argmax = output.detach().permute(1, 0, 2)
            output_argmax = torch.softmax(output_argmax, dim=-1)

            output_2 = model_2(data)
            output_argmax_2 = output_2.detach().permute(1, 0, 2)
            output_argmax_2 = torch.softmax(output_argmax_2, dim=-1)

            output_3 = model_3(data)
            output_argmax_3 = output_3.detach().permute(1, 0, 2)
            output_argmax_3 = torch.softmax(output_argmax_3, dim=-1)

            # ctc_input = (output_argmax + output_argmax_2 + output_argmax_3)/3
            # ctc_input = output_argmax
            # ctc_input = torch.cat([ctc_input[:, :, 1:], ctc_input[:, :, :1]], dim=-1)
            # pred = ctc_decode(ctc_input)

            output_argmax = output_argmax
            # output_argmax = (output_argmax + output_argmax_2 + output_argmax_3)/3
            beam_results, beam_scores, timesteps, out_lens = decoder.decode(output_argmax)
            pred = []
            for idx in range(len(beam_results)):
                idx_4 = 0
                for i in range(len(beam_results[idx])):
                    if out_lens[idx][i] == 4:
                        idx_4 = i
                        break
                
                pred.append(''.join([characters[i] for i in beam_results[idx][idx_4][:out_lens[idx][idx_4]]]))
                # print(beam_scores, beam_scores.shape)
            # output_argmax = (output_argmax + output_argmax_2 + output_argmax_3)/3
            # output_argmax = output_argmax.argmax(dim=-1)
            # output_argmax = output_argmax.cpu().numpy()
            # pred = [decode(p) for p in output_argmax] 
            print(f"{batch_index+1}/{len(dataloader)}", pred)
            res += pred

    with open(submit_file, 'w') as f:
        f.write(f"num,tag\n")
        for idx in range(len(res)):
            l = f"{idx+1},{res[idx]}"
            f.write(l+'\n')


def run():
    batch_size = 16
    valid_set = CaptchaDataset(characters, width, height, n_input_length, n_len, mode='A')
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=12, pin_memory=True)


    model = Model(n_classes, input_shape=(3, height, width))
    model_2 = Model(n_classes, input_shape=(3, height, width))
    model_3 = Model(n_classes, input_shape=(3, height, width), mode='large')
    print(model)

    try:
        checkpoint = torch.load('/new_share/wangqixun/workspace/bs/dcic_ocr/output/checkpoint_latest.pth.tar', map_location='cpu')
        init_epoch = checkpoint['epoch'] + 1
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        print("=> Resume: loaded checkpoint '{}' (epoch {})".format(
            '/new_share/wangqixun/workspace/bs/dcic_ocr/output/checkpoint_latest.pth.tar', checkpoint['epoch']))
        checkpoint = torch.load('/new_share/wangqixun/workspace/bs/dcic_ocr/output/checkpoint_latest.pth.tar.962', map_location='cpu')
        init_epoch = checkpoint['epoch'] + 1
        model_2.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        print("=> Resume: loaded checkpoint '{}' (epoch {})".format(
            '/new_share/wangqixun/workspace/bs/dcic_ocr/output/checkpoint_latest.pth.tar', checkpoint['epoch']))
        checkpoint = torch.load('/new_share/wangqixun/workspace/bs/dcic_ocr/output/checkpoint_latest.pth.tar.9628', map_location='cpu')
        init_epoch = checkpoint['epoch'] + 1
        model_3.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        print("=> Resume: loaded checkpoint '{}' (epoch {})".format(
            '/new_share/wangqixun/workspace/bs/dcic_ocr/output/checkpoint_latest.pth.tar', checkpoint['epoch']))
    except Exception as e:
        print(e)

    model = model.cuda()
    model_2 = model_2.cuda()
    model_3 = model_3.cuda()
    valid([model, model_2, model_3], valid_loader)


if __name__ == '__main__':
    run()


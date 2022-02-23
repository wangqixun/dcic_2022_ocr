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
import cv2
from glob import glob
from ctc_decoder import best_path, beam_search
from ctcdecode import CTCBeamDecoder

import string

from model.convnext import convnext_base, convnext_small, convnext_large

if __name__ == '__main__':
    # 1) 初始化
    torch.distributed.init_process_group(backend="nccl")

    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


print_raw = print
def print(*info):
    if local_rank == 0:
        print_raw(*info)
scaler = GradScaler()


width, height = 320, 128
n_input_length = 80*2
characters = '-' + string.digits + string.ascii_letters
n_len, n_classes = 4*2, len(characters)

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


def save_checkpoint(state, outname='checkpoint_latest.pth.tar'):
    if local_rank == 0:
        best_acc = state['best_acc']
        epoch = state['epoch']
        # filename = 'checkpoint_acc_%.4f_epoch_%02d.pth.tar' % (best_acc, epoch)
        filename = outname
        # filename = 'checkpoint_best_%d.pth.tar'
        os.makedirs('output/', exist_ok=True)
        filename = os.path.join('output/', filename)
        torch.save(state, filename)

        # best_filename = os.path.join(model_dir, 'checkpoint_best_%d.pth.tar' % name_no)
        # best_filename = filename
        # shutil.copyfile(filename, best_filename)
        print('=> Save model to %s' % filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class CaptchaDataset(Dataset):
    def __init__(self, characters, width, height, input_length, label_length, mode='tra'):
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

        if self.mode == 'val':
            self.val_dir = '/new_share/wangqixun/workspace/bs/dcic_ocr/data/training_dataset/'
            self.val_names = open('/new_share/wangqixun/workspace/bs/dcic_ocr/data/val.txt').readlines()
            self.val_names = [t.strip() for t in self.val_names]
            self.transform_val = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            self.tra_dir = '/new_share/wangqixun/workspace/bs/dcic_ocr/data/training_dataset/'
            self.tra_names = open('/new_share/wangqixun/workspace/bs/dcic_ocr/data/tra.txt').readlines()
            self.tra_names = [t.strip() for t in self.tra_names]
            self.transform_tra = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.RandomRotation(degrees=(-5, 5)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])


    def __len__(self):
        if self.mode == 'tra':
            return len(self.tra_names)*3
        else:
            return len(self.val_names)


    def __getitem__(self, index):
        if self.mode == 'tra':
            if index < len(self.tra_names):
                random_str = self.tra_names[index].split('.')[0]
                img_file = os.path.join(self.tra_dir, random_str+'.png')
                image = self.transform_tra(Image.open(img_file).convert('RGB'))
                target = torch.tensor([self.characters.find(x) for x in random_str], dtype=torch.long)

                index_aug2 = np.random.randint(0, len(self.tra_names))
                random_str_aug2 = self.tra_names[index_aug2].split('.')[0]
                img_file = os.path.join(self.tra_dir, random_str_aug2+'.png')
                image_aug2 = self.transform_tra(Image.open(img_file).convert('RGB'))
                image = torch.cat([image, image_aug2], dim=-1)
                target = torch.tensor([self.characters.find(x) for x in random_str+random_str_aug2], dtype=torch.long)

            else:
                random_str = ''.join([random.choice(self.characters[1:]) for j in range(self.label_length//2)])
                width_ic = np.random.randint(110, 140)
                height_ic = int(width_ic/2.5)
                font_dir = '/new_share/wangqixun/workspace/gallery/fonts'
                fonts = list(glob(f"{font_dir}/*.TTF")) + list(glob(f"{font_dir}/*.otf")) + list(glob(f"{font_dir}/*.ttf"))
                image = ImageCaptcha(width=width_ic, height=height_ic, fonts=fonts).generate_image(random_str).convert('RGB')
                # cv2.imwrite(f"/new_share/wangqixun/data/tmp_output/{random_str}.jpg", np.array(image)[..., ::-1])
                image = self.transform_tra(image)
                target = torch.tensor([self.characters.find(x) for x in random_str], dtype=torch.long)

                index_aug2 = np.random.randint(0, len(self.tra_names))
                random_str_aug2 = self.tra_names[index_aug2].split('.')[0]
                img_file = os.path.join(self.tra_dir, random_str_aug2+'.png')
                image_aug2 = self.transform_tra(Image.open(img_file).convert('RGB'))
                image = torch.cat([image, image_aug2], dim=-1)
                target = torch.tensor([self.characters.find(x) for x in random_str+random_str_aug2], dtype=torch.long)



            input_length = torch.full(size=(1, ), fill_value=self.input_length, dtype=torch.long)
            target_length = torch.full(size=(1, ), fill_value=self.label_length, dtype=torch.long)


            return image, target, input_length, target_length
        
        else:
            random_str = self.val_names[index].split('.')[0]
            img_file = os.path.join(self.val_dir, random_str+'.png')
            image = self.transform_val(Image.open(img_file).convert('RGB'))
            target = torch.tensor([self.characters.find(x) for x in random_str], dtype=torch.long)

            # image_aug2 = image
            # image = torch.cat([image, image_aug2], dim=-1)
            # target = torch.tensor([self.characters.find(x) for x in random_str+random_str], dtype=torch.long)

            # random_str = ''.join([random.choice(self.characters[1:]) for j in range(self.label_length)])
            # image = self.transform_val(self.generator.generate_image(random_str).convert('RGB'))
            input_length = torch.full(size=(1, ), fill_value=self.input_length, dtype=torch.long)
            target_length = torch.full(size=(1, ), fill_value=self.label_length, dtype=torch.long)
            return image, target, input_length, target_length


class TransformerEncoder(nn.Module):
    def __init__(self, pretrained_transformers, freeze=False, layers=6, final_channel_number=768):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_transformers)
        self.model.encoder.layer = self.model.encoder.layer[:layers]
        self.model.embeddings.word_embeddings = None
        self.embeddings = self.model.embeddings
        self.model = self.model.encoder
        self.norm = nn.LayerNorm(final_channel_number)
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False   #预训练模型加载进来后全部设置为不更新参数，然后再后面加层

    def forward(self, 
        embedding_input, 
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        embedding_input_with_position = self.embeddings(inputs_embeds=embedding_input)
        output = self.model(
            embedding_input_with_position,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        embedding_output = output[0]

        # [bs, N, C]
        pad_0 = torch.zeros_like(embedding_output)[:, :2, :]
        embedding_output = torch.cat([pad_0, embedding_output, pad_0], dim=1)
        embedding_output = torch.cat([embedding_output[:, (i-2):(i+3), :].reshape(-1, 1, 768*5) for i in range(2, embedding_output.shape[1]-2)], dim=1)

        embedding_output = self.norm(embedding_output)
        return embedding_output


class Model(nn.Module):
    def __init__(self, n_classes, input_shape=(3, 64, 128), mode='base'):
        super(Model, self).__init__()
        self.input_shape = input_shape
        channels = [32, 64, 128, 256, 256]
        layers = [2, 2, 2, 2, 2]
        kernels = [3, 3, 3, 3, 3]
        pools = [2, 2, 2, 2, (2, 2)]
        modules = OrderedDict()
        
        def cba(name, in_channels, out_channels, kernel_size):
            modules[f'conv{name}'] = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(1, 1) if kernel_size == 3 else 0)
            modules[f'bn{name}'] = nn.BatchNorm2d(out_channels)
            modules[f'relu{name}'] = nn.ReLU(inplace=True)
        
        last_channel = 3
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                cba(f'{block+1}{layer}', last_channel, n_channel, n_kernel)
                last_channel = n_channel
            modules[f'pool{block + 1}'] = nn.MaxPool2d(k_pool)
        modules[f'dropout'] = nn.Dropout(0.25, inplace=True)
        
        # self.cnn = nn.Sequential(modules)
        self.backbone = self.get_model(mode)
        self.transformers = nn.Sequential(
            nn.Linear(in_features=self.infer_features(), out_features=768),
            TransformerEncoder(
                pretrained_transformers='/new_share/wangqixun/workspace/githup_project/model_super_strong/transformers/roberta-base', 
                layers=2, 
                final_channel_number=768*5,
            )
        )
        # self.lstm = nn.LSTM(input_size=self.infer_features(), hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=768*5, out_features=n_classes)


    def get_model(self, mode, device='cpu'):
        if mode=='small':
            model = convnext_small(num_classes=1000)
            model_pkl='/new_share/wangqixun/workspace/githup_project/model_super_strong/convnext_small_1k_224_ema.pth' 
        elif mode=='base':
            model = convnext_base(num_classes=1000)
            model_pkl='/new_share/wangqixun/workspace/githup_project/model_super_strong/convnext_base_22k_1k_384.pth'
        elif mode=='large':
            model = convnext_large(num_classes=1000)
            model_pkl='/new_share/wangqixun/workspace/githup_project/model_super_strong/convnext_large_22k_1k_384.pth'

        checkpoint = torch.load(model_pkl, map_location="cpu")
        # model = convnext_large()
        model.load_state_dict(checkpoint['model'])   
        model.to(device)            
        model.eval()
        return model


    def infer_features(self):
        x = torch.zeros((1,)+self.input_shape)
        # x = self.cnn(x)
        x = self.backbone.forward_features_wo_pool(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        # x = torch.mean(x, dim=2)
        return x.shape[1]

    def forward(self, x):
        # print(0, x.shape)
        # x = self.cnn(x)
        x = self.backbone.forward_features_wo_pool(x)   # bs, C, H, W
        # print(1, x.shape)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        # x = torch.mean(x, dim=2)
        # print(2, x.shape)
        # x = x.permute(2, 0, 1)
        # print(3, x.shape)
        # x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.transformers(x)
        # print(4, x.shape)
        x = self.fc(x)
        # print(5, x.shape)
        x = x.permute(1, 0, 2)
        # print(6, x.shape)

        return x


def decode(sequence):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j+1]])
    if len(s) == 0:
        # print('pred:', '')
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    res = s[:4]
    # print('pred:', s, res)
    return res

def decode_target(sequence):
    s = ''.join([characters[x] for x in sequence]).replace(' ', '')
    res = s[:4]
    # print('label:', s, res)
    return res

def calc_acc(target, output):
    # output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    # target = target.cpu().numpy()
    # output_argmax = output_argmax.cpu().numpy()
    # a = np.array([decode_target(true) == decode(pred) for true, pred in zip(target, output_argmax)])
    # return a.mean()
    target = target.cpu().numpy()

    output_argmax = output.detach().permute(1, 0, 2)
    output_argmax = torch.softmax(output_argmax, dim=-1)
    beam_results, beam_scores, timesteps, out_lens = decoder.decode(output_argmax)
    pred = []
    for idx in range(len(beam_results)):

        idx_4 = 0
        for i in range(len(beam_results[idx])):
            # print(target.shape)
            if out_lens[idx][i] == target.shape[1]:
                idx_4 = i
                break
        pred.append(''.join([characters[i] for i in beam_results[idx][idx_4][:out_lens[idx][idx_4]]]))

        # pred.append(''.join([characters[i] for i in beam_results[idx][0][:out_lens[idx][0]]]))

    real = [decode_target(true) for true in target]
    a = np.array([t==p for t, p in zip(real, pred)])
    return a.mean()




def train(model, optimizer, epoch, dataloader, sampler):
    print(f"\n\n=> train")
    data_time = AverageMeter('- data', ':4.3f')
    batch_time = AverageMeter('- batch', ':6.3f')
    losses = AverageMeter('- loss', ':.4e')
    acces = AverageMeter('- acc', ':.4f')
    progress = ProgressMeter(
        len(dataloader), data_time, batch_time, losses, acces, prefix=f"Epoch: [{epoch}]")

    end = time.time()
    model.train()
    sampler.set_epoch(epoch)
    nb_sum = 0
    loss_sum = 0
    acc_sum = 0

    for batch_index, (data, target, input_lengths, target_lengths) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        
        output_log_softmax = F.log_softmax(output, dim=-1)
        loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)
        
        loss.backward()
        optimizer.step()

        loss = loss.item()
        acc = calc_acc(target, output)

        loss_sum += loss
        losses.update(loss, len(data))
        acces.update(acc, len(data))

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % 50 == 0:
            progress.print(batch_index)

        nb_sum += len(data)
        loss_sum += loss * len(data)
        acc_sum += acc * len(data)
        loss_mean = loss_sum / nb_sum
        acc_mean = acc_sum / nb_sum

    return loss_sum, acc_sum


def valid(model, optimizer, epoch, dataloader, sampler):
    print(f"=> val")
    data_time = AverageMeter('- data', ':4.3f')
    batch_time = AverageMeter('- batch', ':6.3f')
    losses = AverageMeter('- loss', ':.4e')
    acces = AverageMeter('- acc', ':.4f')
    progress = ProgressMeter(
        len(dataloader), data_time, batch_time, losses, acces, prefix=f"Epoch: [{epoch}]")

    end = time.time()
    model.eval()
    sampler.set_epoch(epoch)
    nb_sum = 0
    loss_sum = 0
    acc_sum = 0

    with torch.no_grad():
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            
            # optimizer.zero_grad()
            output = model(data)
            
            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)
            # loss.backward()
            # optimizer.step()

            loss = loss.item()
            acc = calc_acc(target, output)

            loss_sum += loss
            losses.update(loss, len(data))
            acces.update(acc, len(data))

            batch_time.update(time.time() - end)
            end = time.time()

            nb_sum += len(data)
            loss_sum += loss * len(data)
            acc_sum += acc * len(data)
            loss_mean = loss_sum / nb_sum
            acc_mean = acc_sum / nb_sum

        progress.print(batch_index)
        
        return loss_sum, acc_sum


def run():

    batch_size = 16
    train_set = CaptchaDataset(characters, width, height, n_input_length, n_len, )
    valid_set = CaptchaDataset(characters, width, height, n_input_length//2, n_len//2, mode='val')
    sampler_tra = DistributedSampler(train_set, shuffle=True)
    sampler_val = DistributedSampler(valid_set, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=12, pin_memory=True, sampler=sampler_tra)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=12, pin_memory=True, sampler=sampler_val)


    model = Model(n_classes, input_shape=(3, height, width))
    # print(model)

    try:
        checkpoint = torch.load('/new_share/wangqixun/workspace/bs/dcic_ocr/output/checkpoint_latest.pth.tar.xxx', map_location='cpu')
        init_epoch = checkpoint['epoch'] + 1
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        optimizer.load_state_dict(checkpoint['optimizer'])
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        print("=> Resume: loaded checkpoint '{}' (epoch {})".format(
            '/new_share/wangqixun/workspace/bs/dcic_ocr/output/checkpoint_latest.pth.tar.xxx', checkpoint['epoch']))
    except Exception as e:
        print(e)

    model = model.cuda()
    # print(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


    loss = 9999
    acc = 0.

    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
    epochs = 40
    for epoch in range(1, epochs + 1):
        loss_tra, acc_tra = train(model, optimizer, epoch, train_loader, sampler_tra)
        loss_val, acc_val = valid(model, optimizer, epoch, valid_loader, sampler_val)
        if acc_val >= acc:
            acc = acc_val
            loss = loss_val
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc': acc,
                'optimizer': optimizer.state_dict(),
            }, outname=f'checkpoint_latest.pth.tar')

    optimizer = torch.optim.AdamW(model.parameters(), 1e-5)
    epochs = 60
    for epoch in range(1, epochs + 1):
        loss_tra, acc_tra = train(model, optimizer, epoch, train_loader, sampler_tra)
        loss_val, acc_val = valid(model, optimizer, epoch, valid_loader, sampler_val)
        if acc_val >= acc:
            acc = acc_val
            loss = loss_val
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc': acc,
                'optimizer': optimizer.state_dict(),
            }, outname=f'checkpoint_latest.pth.tar')



if __name__ == '__main__':
    run()


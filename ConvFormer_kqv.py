import argparse
import os
import random

import numpy as np
import time
from sklearn.metrics import accuracy_score, confusion_matrix
# from models import CNN
from mydataset import MyDataset
import torch
from torch import einsum
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # 导入包
import matplotlib.font_manager as f

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
##ligong2 cccst
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# device=torch.device("cpu")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def draw(train_acc, train_loss, test_acc, test_loss):
    x1 = range(len(train_acc))
    x2 = range(len(train_loss))
    y1 = train_acc
    y2 = train_loss
    y3 = test_acc
    y4 = test_loss
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-', label="train", color='b')
    plt.plot(x1, y3, 'o-', label="test", color='r')
    plt.legend(loc='upper left')
    plt.title('accuracy & NAR vs. epochs')
    plt.ylabel('accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-', label="train", color='b')
    plt.plot(x2, y4, '.-', label="test", color='r')
    plt.legend(loc='upper left')
    plt.xlabel('loss vs. epochs')
    plt.ylabel('loss')
    plt.savefig("ConvFormer_kqv_accuracy_loss.jpg")
    plt.show()


def draw_result(C):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df = pd.DataFrame(C)
    f1 = f.FontProperties(size=15)
    sns.heatmap(df, fmt='g', annot=True,
                annot_kws={'size': 10},
                xticklabels=['1', '2', '3', '4', '5', '6'],
                yticklabels=['1', '2', '3', '4', '5', '6'],
                cmap='YlGnBu')
    ax.set_xlabel('Predicted label', fontproperties=f1)  # xf轴f
    ax.set_ylabel('True label', fontproperties=f1)  # y轴
    plt.savefig('./ConvFormer_kqv_confusion_matrix.jpg')
    plt.show()
    Acc = (C[0][0] + C[1][1] + C[2][2] + C[3][3] + C[4][4] + C[5][5]) / sum(C[0] + C[1] + C[2] + C[3] + C[4] + C[5])
    print('acc: %.3f' % Acc)
    lie_he = sum(C, 1) - 1
    for i in range(1, 7):
        Precision = C[i - 1][i - 1] / lie_he[i - 1]
        NAR = (sum(C[i - 1]) - C[i - 1][i - 1]) / sum(C[i - 1])
        F1_score = 2 * C[i - 1][i - 1] / (lie_he[i - 1] + sum(C[i - 1]))
        print('precision_%d: %.3f' % (i, Precision))
        print('NAR_%d: %.3f' % (i, NAR))
        print('F1_score_%d: %.3f' % (i, F1_score))


def draw_resultv(C):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df = pd.DataFrame(C)
    f1 = f.FontProperties(size=15)
    sns.heatmap(df, fmt='g', annot=True,
                annot_kws={'size': 10},
                xticklabels=['1', '2', '3', '4', '5', '6'],
                yticklabels=['1', '2', '3', '4', '5', '6'],
                cmap='YlGnBu')
    ax.set_xlabel('Predicted label', fontproperties=f1)  # xf轴f
    ax.set_ylabel('True label', fontproperties=f1)  # y轴
    plt.savefig('./ConvFormer_kqv_confusion_matrix.jpg')
    plt.show()
    Acc = (C[0][0] + C[1][1] + C[2][2] + C[3][3] + C[4][4] + C[5][5]) / sum(C[0] + C[1] + C[2] + C[3] + C[4] + C[5])
    print('acc: %.3f' % Acc)
    lie_he = sum(C, 1) - 1
    for i in range(1, 7):
        Precision = C[i - 1][i - 1] / lie_he[i - 1]
        NAR = (sum(C[i - 1]) - C[i - 1][i - 1]) / sum(C[i - 1])
        F1_score = 2 * C[i - 1][i - 1] / (lie_he[i - 1] + sum(C[i - 1]))
        print('precision_%d: %.3f' % (i, Precision))
        print('NAR_%d: %.3f' % (i, NAR))
        print('F1_score_%d: %.3f' % (i, F1_score))


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Transformerkqv(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attqkv = Attentionkqv(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.pre1 = PreNorm(dim, Attentionkqv(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
        self.pre2 = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        self.preNorm1 = PreNorm(dim, Attentionkqv(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.preNorm2 = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x, y):
        x = self.attqkv(x, y) + y
        x = self.ff(x) + x
        return x


class Attentionkqv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        b, n, _, h = *x.shape, self.heads
        # qkv = self.to_qkv(x).chunk(3, dim=-1)           # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        kv = self.to_kv(x).chunk(2, dim=-1)
        q = self.to_q(y)
        q = rearrange(q, 'b n (h d)->b h n d', h=h)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)          # q, k, v   (b, h, n, dim_head(64))
        k, v = map(lambda t: rearrange(t, 'b n (h d) ->b h n d', h=h), kv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class CViT(nn.Module):
    def __init__(self, *, image_size=12, patch_size=1, num_classes=6, dim=128, depth=2, heads=16, mlp_dim=512,
                 pool='cls', channels=128, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.conv2Dblock = nn.Sequential(
            # 1. conv block
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(10, 2), stride=(4, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(p=0.1))

        self.conv2Dblock1 = nn.Sequential(
            # 2. conv block
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(10, 2), stride=(4, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(p=0.1))
        self.conv2Dblock2 = nn.Sequential(
            # 3. conv block
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(12, 2), stride=(4, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(p=0.1))

        self.Timepool0 = nn.AvgPool2d(kernel_size=(1, 12))
        self.Spacepool0 = nn.AvgPool2d(kernel_size=(248, 1))

        transf_layer0 = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=128, dropout=0.1,
                                                   activation='relu', batch_first=True)
        self.transf_encoder0 = nn.TransformerEncoder(transf_layer0, num_layers=4)
        self.l0 = nn.Linear(16, 8)

        transf_layer01 = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=128, dropout=0.1,
                                                    activation='relu', batch_first=True)
        self.transf_encoder01 = nn.TransformerEncoder(transf_layer01, num_layers=4)
        self.l01 = nn.Linear(16, 8)

        self.Timepool = nn.AvgPool2d(kernel_size=(1, 12))
        self.Spacepool = nn.AvgPool2d(kernel_size=(60, 1))

        transf_layer1 = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=128, dropout=0.1,
                                                   activation='relu', batch_first=True)
        self.transf_encoder1 = nn.TransformerEncoder(transf_layer1, num_layers=4)
        self.l1 = nn.Linear(32, 16)

        transf_layer2 = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=128, dropout=0.1,
                                                   activation='relu', batch_first=True)
        self.transf_encoder2 = nn.TransformerEncoder(transf_layer2, num_layers=4)
        self.l2 = nn.Linear(32, 16)

        image_height, image_width = pair(image_size)  # 12 12
        patch_height, patch_width = pair(patch_size)  # 1 1

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 12*12  144
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            # (batch-size,128,12,12) -->()batch-size,144,128)
            nn.Linear(patch_dim, dim)  # 128--->256
        )
        self.tpos_embedding0 = nn.Parameter(torch.randn(1, 248, 16))
        self.spos_embedding0 = nn.Parameter(torch.randn(1, 12, 16))

        self.tpos_embedding = nn.Parameter(torch.randn(1, 60, 32))
        self.spos_embedding = nn.Parameter(torch.randn(1, 12, 32))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformerkqv = Transformerkqv(dim, 1, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        conv_embedding = self.conv2Dblock(x)  # (b,channel,freq,time)(b  ,16 ,248,12)
        # print(conv_embedding.shape)
        Timepooling0 = self.Timepool0(conv_embedding)
        Spacepooling0 = self.Spacepool0(conv_embedding)
        Timepooling0 = torch.squeeze(Timepooling0, 3)  # (b,32,60)
        Spacepooling0 = torch.squeeze(Spacepooling0, 2)  # (b,32,12)
        # print(Timepooling0.shape)
        # print(Spacepooling0.shape)
        Timepooling0 = Timepooling0.permute(0, 2, 1)  # (b,60.32)
        Spacepooling0 = Spacepooling0.permute(0, 2, 1)  # (b,12,32)
        # print(Timepooling.shape)
        # print(Spacepooling.shape)
        Timepooling0 += self.tpos_embedding0[:, :]
        Spacepooling0 += self.spos_embedding0[:, :]
        # print("T:",Timepooling.shape)
        # print("S:",Spacepooling.shape)
        Timepooling0 = self.transf_encoder0(Timepooling0)  # (b,60.32)
        Timepooling0 = self.l0(Timepooling0)
        # print("T:",Timepooling.shape)
        Spacepooling0 = self.transf_encoder01(Spacepooling0)  # (b,12,32)
        Spacepooling0 = self.l01(Spacepooling0)
        # print("S",Spacepooling.shape)
        Timepooling0 = torch.unsqueeze(Timepooling0, 2)  # (b,60,1,32)
        Spacepooling0 = torch.unsqueeze(Spacepooling0, 1)  # (b,1,12,32)
        # print("s:",Spacepooling.shape)
        # print("t:",Timepooling.shape)
        Timepooling0 = Timepooling0.repeat(1, 1, 12, 1)  # (b,60,12,32)
        # print("tt:",Timepooling.shape)
        Spacepooling0 = Spacepooling0.repeat(1, 248, 1, 1)  # (b,60,12,32)
        STvit0 = torch.cat([Timepooling0, Spacepooling0], dim=3)
        STvit0 = STvit0.permute(0, 3, 1, 2)  # (b,64,60,12)
        conv_embedding = conv_embedding + STvit0

        conv_embedding = self.conv2Dblock1(conv_embedding)
        # print("conv_embeffing:",conv_embedding.shape)        ( b,32,60,12)
        # conv_embedding =self.conv2Dblock2(conv_embedding)
        # print(conv_embedding.shape)
        Timepooling = self.Timepool(conv_embedding)  # (b,32,60,1)
        Spacepooling = self.Spacepool(conv_embedding)  # (b,32,1,12)
        # print(Timepooling.shape)
        # print(Spacepooling.shape)
        Timepooling = torch.squeeze(Timepooling, 3)  # (b,32,60)
        Spacepooling = torch.squeeze(Spacepooling, 2)  # (b,32,12)
        # print(Timepooling.shape)
        # print(Spacepooling.shape)
        Timepooling = Timepooling.permute(0, 2, 1)  # (b,60.32)
        Spacepooling = Spacepooling.permute(0, 2, 1)  # (b,12,32)
        # print(Timepooling.shape)
        # print(Spacepooling.shape)
        Timepooling += self.tpos_embedding[:, :]
        Spacepooling += self.spos_embedding[:, :]
        # print("T:",Timepooling.shape)
        # print("S:",Spacepooling.shape)
        Timepooling = self.transf_encoder1(Timepooling)  # (b,60.32)
        Timepooling = self.l1(Timepooling)
        # print("T:",Timepooling.shape)
        Spacepooling = self.transf_encoder2(Spacepooling)  # (b,12,32)
        Spacepooling = self.l2(Spacepooling)
        # print("S",Spacepooling.shape)
        Timepooling = torch.unsqueeze(Timepooling, 2)  # (b,60,1,32)
        Spacepooling = torch.unsqueeze(Spacepooling, 1)  # (b,1,12,32)
        # print("s:",Spacepooling.shape)
        # print("t:",Timepooling.shape)
        Timepooling = Timepooling.repeat(1, 1, 12, 1)  # (b,60,12,32)
        # print("tt:",Timepooling.shape)
        Spacepooling = Spacepooling.repeat(1, 60, 1, 1)  # (b,60,12,32)

        STvit = torch.cat([Timepooling, Spacepooling], dim=3)
        # STvit=torch.cat([Timepooling,Spacepooling],dim=3)
        # print("Stvit:",STvit.shape)
        STvit = STvit.permute(0, 3, 1, 2)  # (b,64,60,12)
        # STvit=torch.cat([STvit,conv_embedding],dim=1)#(b,96,60,12)
        # print("Stvit:",STvit.shape)
        STvit = STvit + conv_embedding
        STvit = self.conv2Dblock2(STvit)  # (b,128,12,12)
        # print("Stvit:",STvit.shape)
        x = self.to_patch_embedding(
            STvit)  # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim   #(batch-size,128,12,12) -->()batch-size,144,128)
        STvit = x
        b, n, _ = x.shape  # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值
        # print(n)
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)  dim=256
        # x = torch.cat((cls_tokens, x), dim=1)               # 将cls_token拼接到patch token中去       (b, 145, dim)
        # print(x.shape)
        x += self.pos_embedding[:, :n]  # 加位置嵌入（直接加）      (b, 145, dim)
        # print(x.shape)
        x = self.dropout(x)

        x = self.transformer(x)                                                 # (b, 145, dim)
        # # x = STvit+x
        x=self.transformerkqv(x,STvit)
        #  x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        #           # (b, dim)
        # print("x:",x.shape)
        #  x=x.view(x.shape[0],-1)
        #  print(x.shape)
        x, _ = torch.max(x, dim=1)
        # x= x[:,-1]
        # x = self.to_latent(x)  # Identity (b, dim)
        #  print("x:",x.shape)

        return x, self.mlp_head(x)




def test(model, dataset, criterion):
    model.eval()
    total_batch_num = 0
    val_loss = 0
    prediction = []
    labels = []
    feature_list = torch.tensor([]).to(device)  # test总数，特征数
    with torch.no_grad():
        for (step, i) in enumerate(dataset):
            # print("test-step:", step)
            total_batch_num = total_batch_num + 1
            batch_x = i['data']
            batch_y = i['label']
            batch_x = torch.unsqueeze(batch_x, dim=1)  # (b, 1, 10000, 12)
            batch_x = batch_x.float()
            if torch.cuda.is_available():
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
            batch_x = batch_x.reshape(batch_x.shape[0], 10, 1000, batch_x.shape[3])
            feature, probs = model(batch_x)  # feature
            batch_label = batch_y.unsqueeze(1).float()
            feature_label = torch.cat((feature, batch_label), dim=1)
            feature_list = torch.cat((feature_list, feature_label), dim=0)  # feature_list
            loss = criterion(probs, batch_y)
            _, pred = torch.max(probs, dim=1)
            predi = pred.tolist()
            label = batch_y.tolist()
            val_loss += loss.item()
            prediction.extend(predi)
            labels.extend(label)
    accuracy = accuracy_score(labels, prediction)
    C = confusion_matrix(labels, prediction)
    return accuracy, val_loss / total_batch_num, feature_list, C



def main(args):
    # 加载测试集
    test_dataset = MyDataset(args.root2, args.txtpath2, transform=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)

    # 初始化模型
    model = CViT()
    if torch.cuda.is_available():
        model = model.to('cuda')
        device = 'cuda'
    else:
        device = 'cpu'

    # 加载之前训练好的模型权重
    model.load_state_dict(torch.load('ConvFormer_kqv.pth', map_location=device))
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 测试模型
    acc_score, loss_score, feature_list, C = test(model, test_loader, criterion)
    print("Test Accuracy: %.8f Test Loss: %.4f" % (acc_score, loss_score))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConvFormer_kqv fou classification")
    '''save model'''
    parser.add_argument("--save", type=str, default="__",
                        help="path to save model")
    '''model parameters'''
    rootpath = 'das_data'
    parser.add_argument("--root", type=str, default=rootpath + '/train',
                        help="rootpath of traindata")
    parser.add_argument("--root2", type=str, default=rootpath + '/test',
                        help="rootpath of testdata")
    parser.add_argument("--root3", type=str, default=rootpath + '/val',
                        help="rootpath of valdata")

    parser.add_argument("--txtpath", type=str, default=rootpath + '/train/label.txt',
                        help="path of train_list")
    parser.add_argument("--txtpath2", type=str, default=rootpath + '/test/label.txt',
                        help="path of test_list")
    parser.add_argument("--txtpath3", type=str, default=rootpath + '/val/label.txt',
                        help="pach of val_list")

    parser.add_argument("--model", type=str, default="CViT",
                        help="type of model to use for classification")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=75,
                        help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    my_args = parser.parse_args()

    main(my_args)

'''(Phi-OTDR2023) Phi-OTDR@ligong1-System-Product-Name:~/Phi-OTDR_dataset_and_code$ python ConvFormer_kqv.py 
epoch-max: 0
lr: 0.0003
Epoch 0 test_accuracy 0.80089343 test_loss 0.5284
Epoch 0 val_accuracy 0.81493299 val_loss 0.5561
epoch-max: 1
lr: 0.0003
Epoch 1 test_accuracy 0.90938098 test_loss 0.2551
Epoch 1 val_accuracy 0.85258456 val_loss 0.3921
epoch-max: 2
lr: 0.0003
Epoch 2 test_accuracy 0.92724952 test_loss 0.1846
Epoch 2 val_accuracy 0.92022974 val_loss 0.2241
epoch-max: 3
lr: 0.0003
Epoch 3 test_accuracy 0.92788768 test_loss 0.2128
Epoch 3 val_accuracy 0.92086790 val_loss 0.2353
epoch-max: 4
lr: 0.0003
Epoch 4 test_accuracy 0.93937460 test_loss 0.1836
Epoch 4 val_accuracy 0.90810466 val_loss 0.2593
epoch-max: 5
lr: 0.0003
Epoch 5 test_accuracy 0.83982131 test_loss 0.4787
Epoch 5 val_accuracy 0.83790683 val_loss 0.4554
epoch-max: 6
lr: 0.0003
Epoch 6 test_accuracy 0.94001276 test_loss 0.1796
Epoch 6 val_accuracy 0.90491385 val_loss 0.2801
epoch-max: 7
lr: 0.0003
Epoch 7 test_accuracy 0.95979579 test_loss 0.1143
Epoch 7 val_accuracy 0.95532865 val_loss 0.1318
epoch-max: 8
lr: 0.0003
Epoch 8 test_accuracy 0.96553925 test_loss 0.1125
Epoch 8 val_accuracy 0.94894703 val_loss 0.1328
epoch-max: 9
lr: 0.0003
Epoch 9 test_accuracy 0.96298660 test_loss 0.0994
Epoch 9 val_accuracy 0.96553925 val_loss 0.1090
epoch-max: 10
lr: 8.999999999999999e-05
Epoch 10 test_accuracy 0.98404595 test_loss 0.0408
Epoch 10 val_accuracy 0.98213146 val_loss 0.0482
epoch-max: 11
lr: 8.999999999999999e-05
Epoch 11 test_accuracy 0.98723676 test_loss 0.0371
Epoch 11 val_accuracy 0.98213146 val_loss 0.0470
epoch-max: 12
lr: 8.999999999999999e-05
Epoch 12 test_accuracy 0.98468411 test_loss 0.0455
Epoch 12 val_accuracy 0.97957881 val_loss 0.0624
epoch-max: 13
lr: 8.999999999999999e-05
Epoch 13 test_accuracy 0.98340779 test_loss 0.0478
Epoch 13 val_accuracy 0.96936822 val_loss 0.0871
epoch-max: 14
lr: 8.999999999999999e-05
Epoch 14 test_accuracy 0.98787492 test_loss 0.0402
Epoch 14 val_accuracy 0.98213146 val_loss 0.0429
epoch-max: 15
lr: 8.999999999999999e-05
Epoch 15 test_accuracy 0.97702616 test_loss 0.0634
Epoch 15 val_accuracy 0.97766433 val_loss 0.0662
epoch-max: 16
lr: 8.999999999999999e-05
Epoch 16 test_accuracy 0.98213146 test_loss 0.0511
Epoch 16 val_accuracy 0.97830249 val_loss 0.0635
epoch-max: 17
lr: 8.999999999999999e-05
Epoch 17 test_accuracy 0.98787492 test_loss 0.0407
Epoch 17 val_accuracy 0.98149330 val_loss 0.0552
epoch-max: 18
lr: 8.999999999999999e-05
Epoch 18 test_accuracy 0.98978941 test_loss 0.0286
Epoch 18 val_accuracy 0.98340779 val_loss 0.0535
epoch-max: 19
lr: 8.999999999999999e-05
Epoch 19 test_accuracy 0.98596043 test_loss 0.0443
Epoch 19 val_accuracy 0.98149330 val_loss 0.0621
epoch-max: 20
lr: 2.6999999999999996e-05
Epoch 20 test_accuracy 0.99170389 test_loss 0.0273
Epoch 20 val_accuracy 0.98021698 val_loss 0.0488
epoch-max: 21
lr: 2.6999999999999996e-05
Epoch 21 test_accuracy 0.99234205 test_loss 0.0264
Epoch 21 val_accuracy 0.98659860 val_loss 0.0370
epoch-max: 22
lr: 2.6999999999999996e-05
Epoch 22 test_accuracy 0.99170389 test_loss 0.0281
Epoch 22 val_accuracy 0.98468411 val_loss 0.0382
epoch-max: 23
lr: 2.6999999999999996e-05
Epoch 23 test_accuracy 0.98978941 test_loss 0.0344
Epoch 23 val_accuracy 0.98532227 val_loss 0.0402
epoch-max: 24
lr: 2.6999999999999996e-05
Epoch 24 test_accuracy 0.99170389 test_loss 0.0333
Epoch 24 val_accuracy 0.98659860 val_loss 0.0374
epoch-max: 25
lr: 2.6999999999999996e-05
Epoch 25 test_accuracy 0.98915124 test_loss 0.0354
Epoch 25 val_accuracy 0.98404595 val_loss 0.0431
epoch-max: 26
lr: 2.6999999999999996e-05
Epoch 26 test_accuracy 0.99170389 test_loss 0.0305
Epoch 26 val_accuracy 0.98787492 val_loss 0.0330
epoch-max: 27
lr: 2.6999999999999996e-05
Epoch 27 test_accuracy 0.99106573 test_loss 0.0347
Epoch 27 val_accuracy 0.98532227 val_loss 0.0400
epoch-max: 28
lr: 2.6999999999999996e-05
Epoch 28 test_accuracy 0.99106573 test_loss 0.0321
Epoch 28 val_accuracy 0.98915124 val_loss 0.0287
epoch-max: 29
lr: 2.6999999999999996e-05
Epoch 29 test_accuracy 0.98596043 test_loss 0.0442
Epoch 29 val_accuracy 0.98787492 val_loss 0.0382
epoch-max: 30
lr: 8.099999999999999e-06
Epoch 30 test_accuracy 0.99042757 test_loss 0.0330
Epoch 30 val_accuracy 0.98659860 val_loss 0.0364
epoch-max: 31
lr: 8.099999999999999e-06
Epoch 31 test_accuracy 0.98915124 test_loss 0.0303
Epoch 31 val_accuracy 0.98851308 val_loss 0.0357
epoch-max: 32
lr: 8.099999999999999e-06
Epoch 32 test_accuracy 0.99106573 test_loss 0.0327
Epoch 32 val_accuracy 0.98851308 val_loss 0.0334
epoch-max: 33
lr: 8.099999999999999e-06
Epoch 33 test_accuracy 0.99298022 test_loss 0.0335
Epoch 33 val_accuracy 0.98851308 val_loss 0.0354
epoch-max: 34
lr: 8.099999999999999e-06
Epoch 34 test_accuracy 0.98915124 test_loss 0.0337
Epoch 34 val_accuracy 0.98532227 val_loss 0.0392
epoch-max: 35
lr: 8.099999999999999e-06
Epoch 35 test_accuracy 0.99106573 test_loss 0.0336
Epoch 35 val_accuracy 0.98723676 val_loss 0.0340
epoch-max: 36
lr: 8.099999999999999e-06
Epoch 36 test_accuracy 0.99106573 test_loss 0.0342
Epoch 36 val_accuracy 0.98851308 val_loss 0.0349
epoch-max: 37
lr: 8.099999999999999e-06
Epoch 37 test_accuracy 0.99361838 test_loss 0.0326
Epoch 37 val_accuracy 0.99042757 val_loss 0.0291
epoch-max: 38
lr: 8.099999999999999e-06
Epoch 38 test_accuracy 0.99106573 test_loss 0.0334
Epoch 38 val_accuracy 0.98915124 val_loss 0.0333
epoch-max: 39
lr: 8.099999999999999e-06
Epoch 39 test_accuracy 0.99234205 test_loss 0.0321
Epoch 39 val_accuracy 0.98851308 val_loss 0.0331
epoch-max: 40
lr: 2.4299999999999996e-06
Epoch 40 test_accuracy 0.99170389 test_loss 0.0322
Epoch 40 val_accuracy 0.98915124 val_loss 0.0334
epoch-max: 41
lr: 2.4299999999999996e-06
Epoch 41 test_accuracy 0.99234205 test_loss 0.0304
Epoch 41 val_accuracy 0.98659860 val_loss 0.0369
epoch-max: 42
lr: 2.4299999999999996e-06
Epoch 42 test_accuracy 0.99298022 test_loss 0.0313
Epoch 42 val_accuracy 0.98915124 val_loss 0.0339
epoch-max: 43
lr: 2.4299999999999996e-06
Epoch 43 test_accuracy 0.99106573 test_loss 0.0326
Epoch 43 val_accuracy 0.98787492 val_loss 0.0353
epoch-max: 44
lr: 2.4299999999999996e-06
Epoch 44 test_accuracy 0.99170389 test_loss 0.0329
Epoch 44 val_accuracy 0.98787492 val_loss 0.0359
epoch-max: 45
lr: 2.4299999999999996e-06
Epoch 45 test_accuracy 0.99106573 test_loss 0.0322
Epoch 45 val_accuracy 0.98723676 val_loss 0.0372
epoch-max: 46
lr: 2.4299999999999996e-06
Epoch 46 test_accuracy 0.99361838 test_loss 0.0305
Epoch 46 val_accuracy 0.98468411 val_loss 0.0376
epoch-max: 47
lr: 2.4299999999999996e-06
Epoch 47 test_accuracy 0.99298022 test_loss 0.0309
Epoch 47 val_accuracy 0.98915124 val_loss 0.0332
epoch-max: 48
lr: 2.4299999999999996e-06
Epoch 48 test_accuracy 0.99234205 test_loss 0.0298
Epoch 48 val_accuracy 0.98659860 val_loss 0.0386
epoch-max: 49
lr: 2.4299999999999996e-06
Epoch 49 test_accuracy 0.99106573 test_loss 0.0296
Epoch 49 val_accuracy 0.98659860 val_loss 0.0348
bestvacc: 0.990427568602425
besttest: 0.9936183790682833
train totally using %.3f seconds  14236.111528635025
train_acc_list: [0.7271197307260779, 0.8845167494790832, 0.9208206443340279, 0.9365282897900304, 0.9476678954960731, 0.9562429876582785, 0.9574451033819522, 0.9628946946626061, 0.9673024523160763, 0.971149222631832, 0.9892610995351819, 0.9937489982368969, 0.9935085750921622, 0.9940695624298765, 0.9939894213816317, 0.9931078698509377, 0.9940695624298765, 0.9944702676711011, 0.9943901266228562, 0.9951915371053054, 0.9977560506491425, 0.9994390126622856, 0.9989581663728162, 0.9992787305657957, 0.9989581663728162, 0.9989581663728162, 0.9987177432280814, 0.9993588716140407, 0.9986376021798365, 0.999118448469306, 0.9993588716140407, 0.9995191537105306, 0.9998397179035102, 1.0, 0.9997595768552653, 0.9996794358070203, 0.9998397179035102, 0.9998397179035102, 0.9995992947587754, 0.9999198589517551, 1.0, 0.9997595768552653, 0.9998397179035102, 0.9998397179035102, 0.9997595768552653, 0.9997595768552653, 0.9996794358070203, 1.0, 0.9999198589517551, 0.9995191537105306]
train_loss_list: [0.7316566431813497, 0.32726564732377816, 0.2301280992943554, 0.18092828726630591, 0.14917391417955064, 0.12949227852205766, 0.11709727725411519, 0.10444740356594936, 0.09418800281801268, 0.08199635950446243, 0.03344103823605073, 0.021801234241054276, 0.021544068792875577, 0.020427305097087192, 0.01987758636532069, 0.021145269661964634, 0.018758929251533978, 0.017293232946942745, 0.01760348113342595, 0.0148438776081313, 0.007651132004976555, 0.004404268298070078, 0.005037045009067021, 0.004253322016932597, 0.004924994143200254, 0.0045596539079325975, 0.004918980166486017, 0.0038275711030318045, 0.004924098861473792, 0.0038597174980144968, 0.0031977938090099124, 0.002707745121692602, 0.0021913939845900274, 0.001808216766894802, 0.0020108645483953185, 0.0017908411311864934, 0.0019004367676524357, 0.001833291505054172, 0.0019302715788555576, 0.0014037445976065501, 0.0012228819361511094, 0.0015593561560107373, 0.0017374510811968016, 0.0014407502032676054, 0.00160381636351654, 0.0014065909166497137, 0.0014283972537425733, 0.0013633027883489847, 0.0013566852499457653, 0.0019486932923449]
test_acc_list: [0.8008934269304403, 0.9093809827696235, 0.9272495213784301, 0.9278876834716018, 0.9393746011486918, 0.839821314613912, 0.9400127632418634, 0.9597957881301851, 0.9655392469687301, 0.9629865985960434, 0.9840459476707084, 0.9872367581365666, 0.98468410976388, 0.9834077855775367, 0.9878749202297383, 0.97702616464582, 0.9821314613911933, 0.9878749202297383, 0.9897894065092534, 0.9859604339502234, 0.9917038927887684, 0.99234205488194, 0.9917038927887684, 0.9897894065092534, 0.9917038927887684, 0.9891512444160817, 0.9917038927887684, 0.9910657306955967, 0.9910657306955967, 0.9859604339502234, 0.990427568602425, 0.9891512444160817, 0.9910657306955967, 0.9929802169751116, 0.9891512444160817, 0.9910657306955967, 0.9910657306955967, 0.9936183790682833, 0.9910657306955967, 0.99234205488194, 0.9917038927887684, 0.99234205488194, 0.9929802169751116, 0.9910657306955967, 0.9917038927887684, 0.9910657306955967, 0.9936183790682833, 0.9929802169751116, 0.99234205488194, 0.9910657306955967]
test_loss_list: [0.5284484620681222, 0.255103473637101, 0.18459054293605137, 0.21283586135571253, 0.18358849134172636, 0.4787289018847276, 0.17964981244853223, 0.11430842764390518, 0.11247877092266988, 0.0994465758956313, 0.0408319181409826, 0.03706584552336218, 0.04546806111406744, 0.04778837117162172, 0.04020856534741458, 0.06336669186730061, 0.05114130251559818, 0.040716993926587154, 0.028638885266204575, 0.04428710447750838, 0.027312075071380122, 0.026366896824305402, 0.028057247037228377, 0.034359758583340574, 0.03330622154242853, 0.035433369105276696, 0.030481048340687045, 0.0346830098569626, 0.032080574125961436, 0.044242604742122445, 0.0330287908604523, 0.03029531437203666, 0.032650881204861024, 0.03346320177958233, 0.033659647644155334, 0.03364900423919756, 0.03423564201667822, 0.03261992471082949, 0.03338701928189447, 0.03210203184121661, 0.032249904991343156, 0.030353934043924544, 0.03133867041647971, 0.03264262637933324, 0.03292060218444355, 0.03222697848646856, 0.03049123693756196, 0.030909824545530255, 0.02977874346024877, 0.02960698296730549]
val_acc_list: [0.8149329929802169, 0.8525845564773452, 0.9202297383535418, 0.9208679004467135, 0.9081046585832802, 0.8379068283343969, 0.9049138481174218, 0.9553286534779835, 0.9489470325462668, 0.9655392469687301, 0.9821314613911933, 0.9821314613911933, 0.9795788130185067, 0.9693682195277601, 0.9821314613911933, 0.9776643267389917, 0.9783024888321634, 0.9814932992980216, 0.9834077855775367, 0.9814932992980216, 0.9802169751116784, 0.9865985960433951, 0.98468410976388, 0.9853222718570517, 0.9865985960433951, 0.9840459476707084, 0.9878749202297383, 0.9853222718570517, 0.9891512444160817, 0.9878749202297383, 0.9865985960433951, 0.98851308232291, 0.98851308232291, 0.98851308232291, 0.9853222718570517, 0.9872367581365666, 0.98851308232291, 0.990427568602425, 0.9891512444160817, 0.98851308232291, 0.9891512444160817, 0.9865985960433951, 0.9891512444160817, 0.9878749202297383, 0.9878749202297383, 0.9872367581365666, 0.98468410976388, 0.9891512444160817, 0.9865985960433951, 0.9865985960433951]
val_loss_list [0.5561495940296017, 0.39212574094191804, 0.22406194359064102, 0.23526576718752634, 0.25930664113043256, 0.4553501521028122, 0.2801159030641429, 0.13177791758611493, 0.13278983234503897, 0.10899883653132283, 0.048178529641852354, 0.04696582706956364, 0.06235013037625396, 0.08708826391167739, 0.042923892711286854, 0.06622593652288315, 0.06345861904205498, 0.05519308494427717, 0.05347727243066884, 0.06209122764370737, 0.0487700728121943, 0.03696927814613743, 0.038233977740562794, 0.040221097075489674, 0.0374021346895238, 0.04313218021647748, 0.03296035334761302, 0.04004328112457689, 0.0286855210502019, 0.03822132729514198, 0.03644291274818266, 0.03568012413523654, 0.03339035031618848, 0.03535722013935922, 0.03919455865059494, 0.03399665836859269, 0.03490058449163798, 0.02911596658995389, 0.03326924076033293, 0.03312320403235891, 0.03342987830873593, 0.036912382149499275, 0.033904878960519066, 0.035287959300909943, 0.03588616677118306, 0.03718063167594219, 0.03757648299825651, 0.03317254371054611, 0.038625902317497615, 0.03481880231362195]
Authorization required, but no authorization protocol specified

/home/Phi-OTDR/Phi-OTDR_dataset_and_code/ConvFormer_kqv.py:58: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
  plt.show()
acc: 0.991
precision_1: 0.994
NAR_1: 0.010
F1_score_1: 0.992
precision_2: 1.000
NAR_2: 0.012
F1_score_2: 0.994
precision_3: 0.981
NAR_3: 0.004
F1_score_3: 0.988
precision_4: 0.991
NAR_4: 0.013
F1_score_4: 0.989
precision_5: 0.996
NAR_5: 0.000
F1_score_5: 0.998
precision_6: 0.984
NAR_6: 0.016
F1_score_6: 0.984
/home/Phi-OTDR/Phi-OTDR_dataset_and_code/ConvFormer_kqv.py:83: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
  plt.show()
acc: 0.987
precision_1: 0.997
NAR_1: 0.006
F1_score_1: 0.995
precision_2: 0.984
NAR_2: 0.008
F1_score_2: 0.988
precision_3: 0.973
NAR_3: 0.000
F1_score_3: 0.986
precision_4: 0.975
NAR_4: 0.021
F1_score_4: 0.977
precision_5: 0.993
NAR_5: 0.011
F1_score_5: 0.991
precision_6: 0.996
NAR_6: 0.037
F1_score_6: 0.979

'''

'''new 
epoch-max: 0
lr: 0.0003
Epoch 0 test_accuracy 0.86662412 test_loss 0.3842
Epoch 0 val_accuracy 0.85896618 val_loss 0.4071
epoch-max: 1
lr: 0.0003
Epoch 1 test_accuracy 0.92724952 test_loss 0.2157
Epoch 1 val_accuracy 0.92405871 val_loss 0.2212
epoch-max: 2
lr: 0.0003
Epoch 2 test_accuracy 0.93299298 test_loss 0.2120
Epoch 2 val_accuracy 0.92788768 val_loss 0.1991
epoch-max: 3
lr: 0.0003
Epoch 3 test_accuracy 0.93937460 test_loss 0.1695
Epoch 3 val_accuracy 0.92724952 val_loss 0.2068
epoch-max: 4
lr: 0.0003
Epoch 4 test_accuracy 0.97255903 test_loss 0.0881
Epoch 4 val_accuracy 0.94830887 val_loss 0.1340
epoch-max: 5
lr: 0.0003
Epoch 5 test_accuracy 0.91831525 test_loss 0.2320
Epoch 5 val_accuracy 0.94065093 val_loss 0.1722
epoch-max: 6
lr: 0.0003
Epoch 6 test_accuracy 0.93746011 test_loss 0.1803
Epoch 6 val_accuracy 0.93171666 val_loss 0.2061
epoch-max: 7
lr: 0.0003
Epoch 7 test_accuracy 0.94320357 test_loss 0.1533
Epoch 7 val_accuracy 0.94958519 val_loss 0.1565
epoch-max: 8
lr: 0.0003
Epoch 8 test_accuracy 0.94447990 test_loss 0.1679
Epoch 8 val_accuracy 0.94447990 val_loss 0.1585
epoch-max: 9
lr: 0.0003
Epoch 9 test_accuracy 0.91895341 test_loss 0.2569
Epoch 9 val_accuracy 0.92214422 val_loss 0.2275
epoch-max: 10
lr: 8.999999999999999e-05
Epoch 10 test_accuracy 0.98340779 test_loss 0.0546
Epoch 10 val_accuracy 0.98085514 val_loss 0.0512
epoch-max: 11
lr: 8.999999999999999e-05
Epoch 11 test_accuracy 0.97064454 test_loss 0.0914
Epoch 11 val_accuracy 0.98149330 val_loss 0.0539
epoch-max: 12
lr: 8.999999999999999e-05
Epoch 12 test_accuracy 0.98596043 test_loss 0.0533
Epoch 12 val_accuracy 0.98787492 val_loss 0.0401
epoch-max: 13
lr: 8.999999999999999e-05
Epoch 13 test_accuracy 0.97383535 test_loss 0.0862
Epoch 13 val_accuracy 0.96873006 val_loss 0.1011
epoch-max: 14
lr: 8.999999999999999e-05
Epoch 14 test_accuracy 0.98340779 test_loss 0.0447
Epoch 14 val_accuracy 0.98723676 val_loss 0.0485
epoch-max: 15
lr: 8.999999999999999e-05
Epoch 15 test_accuracy 0.98213146 test_loss 0.0535
Epoch 15 val_accuracy 0.98468411 val_loss 0.0548
epoch-max: 16
lr: 8.999999999999999e-05
Epoch 16 test_accuracy 0.97192087 test_loss 0.1153
Epoch 16 val_accuracy 0.97702616 val_loss 0.0778
epoch-max: 17
lr: 8.999999999999999e-05
Epoch 17 test_accuracy 0.98596043 test_loss 0.0481
Epoch 17 val_accuracy 0.98213146 val_loss 0.0548
epoch-max: 18
lr: 8.999999999999999e-05
Epoch 18 test_accuracy 0.90172304 test_loss 0.3405
Epoch 18 val_accuracy 0.91384812 val_loss 0.3030
epoch-max: 19
lr: 8.999999999999999e-05
Epoch 19 test_accuracy 0.98659860 test_loss 0.0462
Epoch 19 val_accuracy 0.98085514 val_loss 0.0642
epoch-max: 20
lr: 2.6999999999999996e-05
Epoch 20 test_accuracy 0.98659860 test_loss 0.0444
Epoch 20 val_accuracy 0.98851308 val_loss 0.0419
epoch-max: 21
lr: 2.6999999999999996e-05
Epoch 21 test_accuracy 0.99106573 test_loss 0.0362
Epoch 21 val_accuracy 0.98915124 val_loss 0.0422
epoch-max: 22
lr: 2.6999999999999996e-05
Epoch 22 test_accuracy 0.99042757 test_loss 0.0360
Epoch 22 val_accuracy 0.98723676 val_loss 0.0457
epoch-max: 23
lr: 2.6999999999999996e-05
Epoch 23 test_accuracy 0.98787492 test_loss 0.0373
Epoch 23 val_accuracy 0.98915124 val_loss 0.0379
epoch-max: 24
lr: 2.6999999999999996e-05
Epoch 24 test_accuracy 0.98915124 test_loss 0.0318
Epoch 24 val_accuracy 0.98915124 val_loss 0.0433
epoch-max: 25
lr: 2.6999999999999996e-05
Epoch 25 test_accuracy 0.98978941 test_loss 0.0370
Epoch 25 val_accuracy 0.98596043 val_loss 0.0447
epoch-max: 26
lr: 2.6999999999999996e-05
Epoch 26 test_accuracy 0.98851308 test_loss 0.0461
Epoch 26 val_accuracy 0.98851308 val_loss 0.0478
epoch-max: 27
lr: 2.6999999999999996e-05
Epoch 27 test_accuracy 0.98723676 test_loss 0.0414
Epoch 27 val_accuracy 0.98978941 val_loss 0.0449
epoch-max: 28
lr: 2.6999999999999996e-05
Epoch 28 test_accuracy 0.98915124 test_loss 0.0423
Epoch 28 val_accuracy 0.98915124 val_loss 0.0456
epoch-max: 29
lr: 2.6999999999999996e-05
Epoch 29 test_accuracy 0.98149330 test_loss 0.0589
Epoch 29 val_accuracy 0.98659860 val_loss 0.0495
epoch-max: 30
lr: 8.099999999999999e-06
Epoch 30 test_accuracy 0.98787492 test_loss 0.0453
Epoch 30 val_accuracy 0.98978941 val_loss 0.0395
epoch-max: 31
lr: 8.099999999999999e-06
Epoch 31 test_accuracy 0.98851308 test_loss 0.0438
Epoch 31 val_accuracy 0.98915124 val_loss 0.0390
epoch-max: 32
lr: 8.099999999999999e-06
Epoch 32 test_accuracy 0.98787492 test_loss 0.0431
Epoch 32 val_accuracy 0.98978941 val_loss 0.0363
epoch-max: 33
lr: 8.099999999999999e-06
Epoch 33 test_accuracy 0.98915124 test_loss 0.0435
Epoch 33 val_accuracy 0.98787492 val_loss 0.0388
epoch-max: 34
lr: 8.099999999999999e-06
Epoch 34 test_accuracy 0.98978941 test_loss 0.0404
Epoch 34 val_accuracy 0.98915124 val_loss 0.0382
epoch-max: 35
lr: 8.099999999999999e-06
Epoch 35 test_accuracy 0.98787492 test_loss 0.0424
Epoch 35 val_accuracy 0.98915124 val_loss 0.0423
epoch-max: 36
lr: 8.099999999999999e-06
Epoch 36 test_accuracy 0.98851308 test_loss 0.0417
Epoch 36 val_accuracy 0.98978941 val_loss 0.0438
epoch-max: 37
lr: 8.099999999999999e-06
Epoch 37 test_accuracy 0.98723676 test_loss 0.0426
Epoch 37 val_accuracy 0.99042757 val_loss 0.0443
epoch-max: 38
lr: 8.099999999999999e-06
Epoch 38 test_accuracy 0.99042757 test_loss 0.0380
Epoch 38 val_accuracy 0.98978941 val_loss 0.0386
epoch-max: 39
lr: 8.099999999999999e-06
Epoch 39 test_accuracy 0.98851308 test_loss 0.0429
Epoch 39 val_accuracy 0.98978941 val_loss 0.0392
epoch-max: 40
lr: 2.4299999999999996e-06
Epoch 40 test_accuracy 0.98915124 test_loss 0.0395
Epoch 40 val_accuracy 0.98978941 val_loss 0.0392
epoch-max: 41
lr: 2.4299999999999996e-06
Epoch 41 test_accuracy 0.98915124 test_loss 0.0406
Epoch 41 val_accuracy 0.99106573 val_loss 0.0368
epoch-max: 42
lr: 2.4299999999999996e-06
Epoch 42 test_accuracy 0.98851308 test_loss 0.0421
Epoch 42 val_accuracy 0.98978941 val_loss 0.0398
epoch-max: 43
lr: 2.4299999999999996e-06
Epoch 43 test_accuracy 0.98787492 test_loss 0.0443
Epoch 43 val_accuracy 0.98915124 val_loss 0.0411
epoch-max: 44
lr: 2.4299999999999996e-06
Epoch 44 test_accuracy 0.98978941 test_loss 0.0428
Epoch 44 val_accuracy 0.98915124 val_loss 0.0391
epoch-max: 45
lr: 2.4299999999999996e-06
Epoch 45 test_accuracy 0.98915124 test_loss 0.0392
Epoch 45 val_accuracy 0.98915124 val_loss 0.0399
epoch-max: 46
lr: 2.4299999999999996e-06
Epoch 46 test_accuracy 0.99042757 test_loss 0.0381
Epoch 46 val_accuracy 0.98915124 val_loss 0.0396
epoch-max: 47
lr: 2.4299999999999996e-06
Epoch 47 test_accuracy 0.98978941 test_loss 0.0389
Epoch 47 val_accuracy 0.98915124 val_loss 0.0388
epoch-max: 48
lr: 2.4299999999999996e-06
Epoch 48 test_accuracy 0.98851308 test_loss 0.0406
Epoch 48 val_accuracy 0.98978941 val_loss 0.0377
epoch-max: 49
lr: 2.4299999999999996e-06
Epoch 49 test_accuracy 0.98978941 test_loss 0.0379
Epoch 49 val_accuracy 0.98915124 val_loss 0.0404
bestvacc: 0.9910657306955967
besttest: 0.9891512444160817
train totally using %.3f seconds  13442.32149887085
train_acc_list: [0.7385799006251001, 0.8951755088956563, 0.9258695303734573, 0.940535342202276, 0.9541593204039109, 0.9567238339477481, 0.9663407597371374, 0.9699471069081583, 0.9666613239301171, 0.9764385318159962, 0.9897419458246514, 0.9947908318640808, 0.9954319602500401, 0.9943901266228562, 0.9967943580702036, 0.9952716781535502, 0.9955121012982849, 0.9947908318640808, 0.9971950633114282, 0.9939092803333868, 0.9986376021798365, 0.9993588716140407, 0.999118448469306, 0.9992787305657957, 0.999118448469306, 0.9993588716140407, 0.9995191537105306, 0.999118448469306, 0.9997595768552653, 0.9994390126622856, 0.9995992947587754, 0.9998397179035102, 0.9998397179035102, 1.0, 1.0, 1.0, 0.9997595768552653, 0.9998397179035102, 0.9999198589517551, 0.9995992947587754, 0.9995992947587754, 1.0, 0.9998397179035102, 0.9999198589517551, 0.9999198589517551, 0.9998397179035102, 1.0, 0.9998397179035102, 1.0, 0.9998397179035102]
train_loss_list: [0.6893418625495121, 0.29613163457240116, 0.21310685165130694, 0.1632068869370517, 0.13500130290026682, 0.11874397314970016, 0.09144392901442581, 0.08552072835178773, 0.08926134357518567, 0.06864545558660667, 0.02965559874323692, 0.020178517069042706, 0.015441141307358762, 0.01662507159997539, 0.013161372735882106, 0.015550284541557977, 0.014049627055557071, 0.01601806518184512, 0.009808935010846127, 0.020840105326001706, 0.0051635801839653595, 0.003888654766763216, 0.004307479098261751, 0.0034161855438308176, 0.004401893003312224, 0.002835860194922377, 0.0029652839932266914, 0.003575585738535501, 0.0023116182820490804, 0.0026261107894354967, 0.0022039169001295458, 0.0015179071369506706, 0.0017678146735173386, 0.0011952600865823859, 0.0015632331255789856, 0.0011807876383156425, 0.0014713016856904083, 0.0013606796553403326, 0.0012184944166771944, 0.0015523868603518132, 0.0014307143621249626, 0.0010281268880841115, 0.0011920930889846905, 0.0009929115948834992, 0.0011608300230321882, 0.0010158772261081273, 0.0010372300358542426, 0.0010853180798563082, 0.0007882388432279462, 0.0010695126494449615]
test_acc_list: [0.8666241225271218, 0.9272495213784301, 0.9329929802169751, 0.9393746011486918, 0.9725590299936184, 0.9183152520740268, 0.9374601148691768, 0.9432035737077218, 0.9444798978940651, 0.9189534141671984, 0.9834077855775367, 0.9706445437141034, 0.9859604339502234, 0.9738353541799617, 0.9834077855775367, 0.9821314613911933, 0.9719208679004467, 0.9859604339502234, 0.9017230376515635, 0.9865985960433951, 0.9865985960433951, 0.9910657306955967, 0.990427568602425, 0.9878749202297383, 0.9891512444160817, 0.9897894065092534, 0.98851308232291, 0.9872367581365666, 0.9891512444160817, 0.9814932992980216, 0.9878749202297383, 0.98851308232291, 0.9878749202297383, 0.9891512444160817, 0.9897894065092534, 0.9878749202297383, 0.98851308232291, 0.9872367581365666, 0.990427568602425, 0.98851308232291, 0.9891512444160817, 0.9891512444160817, 0.98851308232291, 0.9878749202297383, 0.9897894065092534, 0.9891512444160817, 0.990427568602425, 0.9897894065092534, 0.98851308232291, 0.9897894065092534]
test_loss_list: [0.38415624547217575, 0.2156728052401117, 0.2119763181829939, 0.16947689274211927, 0.08809836721285341, 0.23199707489432198, 0.18034755861462684, 0.1532529471501499, 0.16790885554284465, 0.2569374754948884, 0.05457215381302747, 0.09143230261676707, 0.05334450297202554, 0.08619825435655039, 0.0447130797321585, 0.05346596642303499, 0.11530072560025455, 0.04808878744667282, 0.34049657930139504, 0.04618291832861248, 0.04439937249874359, 0.03624864304167984, 0.03597137990184318, 0.03731865581164816, 0.03179645121001938, 0.03696609763603786, 0.04607054636631356, 0.041430610245864655, 0.04225370103843173, 0.05892073052679307, 0.04528211496891549, 0.0437528840656694, 0.043148199485734284, 0.043492485548215755, 0.04039652164268476, 0.042350001536723705, 0.041710693522223403, 0.04255963599332192, 0.03799002988196254, 0.04290547060046098, 0.03946536008360064, 0.040558812425193276, 0.042053634258360145, 0.04428881168964186, 0.04276655242656303, 0.039233813553214626, 0.03813639090760856, 0.03889045150735539, 0.04063823272839987, 0.03785251012811145]
val_acc_list: [0.8589661774090619, 0.9240587109125717, 0.9278876834716018, 0.9272495213784301, 0.9483088704530951, 0.9406509253350351, 0.9317166560306318, 0.9495851946394385, 0.9444798978940651, 0.9221442246330568, 0.9808551372048501, 0.9814932992980216, 0.9878749202297383, 0.9687300574345884, 0.9872367581365666, 0.98468410976388, 0.97702616464582, 0.9821314613911933, 0.9138481174218251, 0.9808551372048501, 0.98851308232291, 0.9891512444160817, 0.9872367581365666, 0.9891512444160817, 0.9891512444160817, 0.9859604339502234, 0.98851308232291, 0.9897894065092534, 0.9891512444160817, 0.9865985960433951, 0.9897894065092534, 0.9891512444160817, 0.9897894065092534, 0.9878749202297383, 0.9891512444160817, 0.9891512444160817, 0.9897894065092534, 0.990427568602425, 0.9897894065092534, 0.9897894065092534, 0.9897894065092534, 0.9910657306955967, 0.9897894065092534, 0.9891512444160817, 0.9891512444160817, 0.9891512444160817, 0.9891512444160817, 0.9891512444160817, 0.9897894065092534, 0.9891512444160817]
val_loss_list [0.40712969575305374, 0.22121303605523948, 0.19905105029822004, 0.2067650024225574, 0.13396362256857433, 0.1722177368237124, 0.2061035342381469, 0.15649870430043308, 0.15849807782678352, 0.22752880259406305, 0.051202146332278584, 0.05389522239464877, 0.040068780424546126, 0.10106981816352345, 0.04849161038160971, 0.05484671822846487, 0.07775723960337333, 0.0548276864062004, 0.30296717032823445, 0.06415639924047614, 0.041862735762355885, 0.042200657808280084, 0.045707706760553336, 0.03786292151525398, 0.04331519819064926, 0.044696903905305035, 0.047774343590186175, 0.04490975742819315, 0.04557404443574119, 0.04946836638607367, 0.03953741851743259, 0.039049456903517334, 0.03626925530120296, 0.03879500118436765, 0.0381836852411281, 0.042254403214076086, 0.04384575805804106, 0.04426046999666439, 0.03859004242481113, 0.03918127531921657, 0.03919455281052529, 0.03679486119401598, 0.039761989117286414, 0.04110769147674936, 0.03914950012107028, 0.039938528462086224, 0.039624482256061, 0.03883812726803162, 0.03767596180025103, 0.040424206810563856]
qt.qpa.plugin: Could not find the Qt platform plugin "xcb" in ""
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
'''

'''new2
epoch-max: 0
lr: 0.0003
Epoch 0 test_accuracy 0.89597958 test_loss 0.2980
Epoch 0 val_accuracy 0.83407786 val_loss 0.4203
epoch-max: 1
lr: 0.0003
Epoch 1 test_accuracy 0.90874282 test_loss 0.2545
Epoch 1 val_accuracy 0.90555201 val_loss 0.2672
epoch-max: 2
lr: 0.0003
Epoch 2 test_accuracy 0.93554563 test_loss 0.1911
Epoch 2 val_accuracy 0.92342055 val_loss 0.2143
epoch-max: 3
lr: 0.0003
Epoch 3 test_accuracy 0.93937460 test_loss 0.1730
Epoch 3 val_accuracy 0.92788768 val_loss 0.2073
epoch-max: 4
lr: 0.0003
Epoch 4 test_accuracy 0.89087428 test_loss 0.2785
Epoch 4 val_accuracy 0.85130823 val_loss 0.4239
epoch-max: 5
lr: 0.0003
Epoch 5 test_accuracy 0.96490108 test_loss 0.0976
Epoch 5 val_accuracy 0.95979579 val_loss 0.1061
epoch-max: 6
lr: 0.0003
Epoch 6 test_accuracy 0.91448628 test_loss 0.2459
Epoch 6 val_accuracy 0.90619017 val_loss 0.2704
epoch-max: 7
lr: 0.0003
Epoch 7 test_accuracy 0.96490108 test_loss 0.1152
Epoch 7 val_accuracy 0.94447990 val_loss 0.1662
epoch-max: 8
lr: 0.0003
Epoch 8 test_accuracy 0.96490108 test_loss 0.1090
Epoch 8 val_accuracy 0.95532865 val_loss 0.1158
epoch-max: 9
lr: 0.0003
Epoch 9 test_accuracy 0.96681557 test_loss 0.1021
Epoch 9 val_accuracy 0.96936822 val_loss 0.0906
epoch-max: 10
lr: 8.999999999999999e-05
Epoch 10 test_accuracy 0.98659860 test_loss 0.0405
Epoch 10 val_accuracy 0.98468411 val_loss 0.0478
epoch-max: 11
lr: 8.999999999999999e-05
Epoch 11 test_accuracy 0.98276962 test_loss 0.0587
Epoch 11 val_accuracy 0.98149330 val_loss 0.0552
epoch-max: 12
lr: 8.999999999999999e-05
Epoch 12 test_accuracy 0.98085514 test_loss 0.0521
Epoch 12 val_accuracy 0.98276962 val_loss 0.0541
epoch-max: 13
lr: 8.999999999999999e-05
Epoch 13 test_accuracy 0.98468411 test_loss 0.0484
Epoch 13 val_accuracy 0.97447352 val_loss 0.0759
epoch-max: 14
lr: 8.999999999999999e-05
Epoch 14 test_accuracy 0.98149330 test_loss 0.0604
Epoch 14 val_accuracy 0.98659860 val_loss 0.0486
epoch-max: 15
lr: 8.999999999999999e-05
Epoch 15 test_accuracy 0.97255903 test_loss 0.0717
Epoch 15 val_accuracy 0.97574984 val_loss 0.0688
epoch-max: 16
lr: 8.999999999999999e-05
Epoch 16 test_accuracy 0.97766433 test_loss 0.0839
Epoch 16 val_accuracy 0.97064454 val_loss 0.0912
epoch-max: 17
lr: 8.999999999999999e-05
Epoch 17 test_accuracy 0.97957881 test_loss 0.0563
Epoch 17 val_accuracy 0.97894065 val_loss 0.0539
epoch-max: 18
lr: 8.999999999999999e-05
Epoch 18 test_accuracy 0.98468411 test_loss 0.0491
Epoch 18 val_accuracy 0.97638800 val_loss 0.0748
epoch-max: 19
lr: 8.999999999999999e-05
Epoch 19 test_accuracy 0.98468411 test_loss 0.0456
Epoch 19 val_accuracy 0.98468411 val_loss 0.0417
epoch-max: 20
lr: 2.6999999999999996e-05
Epoch 20 test_accuracy 0.98978941 test_loss 0.0294
Epoch 20 val_accuracy 0.98915124 val_loss 0.0362
epoch-max: 21
lr: 2.6999999999999996e-05
Epoch 21 test_accuracy 0.98787492 test_loss 0.0339
Epoch 21 val_accuracy 0.98723676 val_loss 0.0381
epoch-max: 22
lr: 2.6999999999999996e-05
Epoch 22 test_accuracy 0.98978941 test_loss 0.0318
Epoch 22 val_accuracy 0.98851308 val_loss 0.0329
epoch-max: 23
lr: 2.6999999999999996e-05
Epoch 23 test_accuracy 0.98978941 test_loss 0.0353
Epoch 23 val_accuracy 0.98915124 val_loss 0.0373
epoch-max: 24
lr: 2.6999999999999996e-05
Epoch 24 test_accuracy 0.99106573 test_loss 0.0350
Epoch 24 val_accuracy 0.98596043 val_loss 0.0405
epoch-max: 25
lr: 2.6999999999999996e-05
Epoch 25 test_accuracy 0.99042757 test_loss 0.0355
Epoch 25 val_accuracy 0.98723676 val_loss 0.0364
epoch-max: 26
lr: 2.6999999999999996e-05
Epoch 26 test_accuracy 0.98851308 test_loss 0.0350
Epoch 26 val_accuracy 0.98787492 val_loss 0.0329
epoch-max: 27
lr: 2.6999999999999996e-05
Epoch 27 test_accuracy 0.99042757 test_loss 0.0331
Epoch 27 val_accuracy 0.99042757 val_loss 0.0320
epoch-max: 28
lr: 2.6999999999999996e-05
Epoch 28 test_accuracy 0.98723676 test_loss 0.0367
Epoch 28 val_accuracy 0.98723676 val_loss 0.0370
epoch-max: 29
lr: 2.6999999999999996e-05
Epoch 29 test_accuracy 0.98787492 test_loss 0.0462
Epoch 29 val_accuracy 0.98404595 val_loss 0.0483
epoch-max: 30
lr: 8.099999999999999e-06
Epoch 30 test_accuracy 0.98915124 test_loss 0.0331
Epoch 30 val_accuracy 0.98978941 val_loss 0.0376
epoch-max: 31
lr: 8.099999999999999e-06
Epoch 31 test_accuracy 0.98915124 test_loss 0.0322
Epoch 31 val_accuracy 0.98787492 val_loss 0.0356
epoch-max: 32
lr: 8.099999999999999e-06
Epoch 32 test_accuracy 0.98978941 test_loss 0.0326
Epoch 32 val_accuracy 0.98978941 val_loss 0.0334
epoch-max: 33
lr: 8.099999999999999e-06
Epoch 33 test_accuracy 0.99106573 test_loss 0.0325
Epoch 33 val_accuracy 0.98787492 val_loss 0.0358
epoch-max: 34
lr: 8.099999999999999e-06
Epoch 34 test_accuracy 0.98978941 test_loss 0.0337
Epoch 34 val_accuracy 0.98915124 val_loss 0.0322
epoch-max: 35
lr: 8.099999999999999e-06
Epoch 35 test_accuracy 0.99042757 test_loss 0.0324
Epoch 35 val_accuracy 0.98851308 val_loss 0.0366
epoch-max: 36
lr: 8.099999999999999e-06
Epoch 36 test_accuracy 0.99042757 test_loss 0.0319
Epoch 36 val_accuracy 0.98787492 val_loss 0.0341
epoch-max: 37
lr: 8.099999999999999e-06
Epoch 37 test_accuracy 0.98978941 test_loss 0.0330
Epoch 37 val_accuracy 0.98915124 val_loss 0.0329
epoch-max: 38
lr: 8.099999999999999e-06
Epoch 38 test_accuracy 0.99042757 test_loss 0.0300
Epoch 38 val_accuracy 0.99042757 val_loss 0.0321
epoch-max: 39
lr: 8.099999999999999e-06
Epoch 39 test_accuracy 0.98978941 test_loss 0.0357
Epoch 39 val_accuracy 0.98851308 val_loss 0.0360
epoch-max: 40
lr: 2.4299999999999996e-06
Epoch 40 test_accuracy 0.99042757 test_loss 0.0331
Epoch 40 val_accuracy 0.98851308 val_loss 0.0370
epoch-max: 41
lr: 2.4299999999999996e-06
Epoch 41 test_accuracy 0.98978941 test_loss 0.0331
Epoch 41 val_accuracy 0.98978941 val_loss 0.0323
epoch-max: 42
lr: 2.4299999999999996e-06
Epoch 42 test_accuracy 0.98978941 test_loss 0.0319
Epoch 42 val_accuracy 0.98851308 val_loss 0.0328
epoch-max: 43
lr: 2.4299999999999996e-06
Epoch 43 test_accuracy 0.98978941 test_loss 0.0339
Epoch 43 val_accuracy 0.98851308 val_loss 0.0337
epoch-max: 44
lr: 2.4299999999999996e-06
Epoch 44 test_accuracy 0.98915124 test_loss 0.0356
Epoch 44 val_accuracy 0.98978941 val_loss 0.0337
epoch-max: 45
lr: 2.4299999999999996e-06
Epoch 45 test_accuracy 0.98978941 test_loss 0.0331
Epoch 45 val_accuracy 0.98851308 val_loss 0.0349
epoch-max: 46
lr: 2.4299999999999996e-06
Epoch 46 test_accuracy 0.98915124 test_loss 0.0341
Epoch 46 val_accuracy 0.99042757 val_loss 0.0351
epoch-max: 47
lr: 2.4299999999999996e-06
Epoch 47 test_accuracy 0.99106573 test_loss 0.0316
Epoch 47 val_accuracy 0.99042757 val_loss 0.0338
epoch-max: 48
lr: 2.4299999999999996e-06
Epoch 48 test_accuracy 0.98915124 test_loss 0.0334
Epoch 48 val_accuracy 0.99106573 val_loss 0.0326
epoch-max: 49
lr: 2.4299999999999996e-06
Epoch 49 test_accuracy 0.99042757 test_loss 0.0346
Epoch 49 val_accuracy 0.99106573 val_loss 0.0375
bestvacc: 0.9910657306955967
besttest: 0.990427568602425
train totally using %.3f seconds  13376.937732219696
train_acc_list: [0.7259977560506491, 0.8889245071325533, 0.9198589517550889, 0.9403750601057862, 0.9507933963776246, 0.9600897579740343, 0.96393652828979, 0.9652989261099535, 0.9705080942458727, 0.9736335951274243, 0.9926270235614681, 0.9946305497675909, 0.9947908318640808, 0.9940695624298765, 0.9949511139605706, 0.9951915371053054, 0.9949511139605706, 0.9932681519474275, 0.9947106908158359, 0.9959128065395095, 0.9989581663728162, 0.9988780253245713, 0.9983971790351018, 0.9994390126622856, 0.999038307421061, 0.999038307421061, 0.9991985895175509, 0.9993588716140407, 0.9992787305657957, 0.9986376021798365, 0.9993588716140407, 0.9997595768552653, 0.9996794358070203, 0.9991985895175509, 0.9997595768552653, 0.9998397179035102, 0.9998397179035102, 0.9995992947587754, 0.9999198589517551, 0.9995191537105306, 0.9995992947587754, 0.9998397179035102, 0.9999198589517551, 1.0, 0.9998397179035102, 0.9997595768552653, 0.9999198589517551, 0.9998397179035102, 0.9997595768552653, 0.9998397179035102]
train_loss_list: [0.7277856366876465, 0.31213070277919497, 0.2241678776166816, 0.16894881268795628, 0.13879881640249783, 0.11521950775666455, 0.10290114592598, 0.09582910894902284, 0.08410684307103446, 0.07848349336359967, 0.02654858942683499, 0.021070600258351654, 0.01814183437416287, 0.020065252513835414, 0.015790616020866715, 0.016437344082480338, 0.0156116699017309, 0.021463883762393905, 0.01621089990632034, 0.013920818664481298, 0.00513690926610722, 0.004603277482101358, 0.005636976064891485, 0.003548500707174309, 0.0046092402534128425, 0.004009384813978562, 0.0037836260172394213, 0.0036524200804051547, 0.003123289559646701, 0.005234960047984532, 0.0026646166064481403, 0.0022566596853814944, 0.001988468837150993, 0.0026603175771520247, 0.0018869703388864182, 0.0015491199009910533, 0.0015471116582251628, 0.0019603509245384103, 0.0013744037693838055, 0.0017252327142180672, 0.001977030215695875, 0.0014747296746746592, 0.0012673971854719802, 0.0010088630019907676, 0.001407760226614097, 0.0015745481538585121, 0.0012346774076768807, 0.001398057991438719, 0.001295882909581757, 0.0013246535806541672]
test_acc_list: [0.8959795788130185, 0.9087428206764518, 0.9355456285896617, 0.9393746011486918, 0.8908742820676452, 0.9649010848755584, 0.9144862795149968, 0.9649010848755584, 0.9649010848755584, 0.9668155711550734, 0.9865985960433951, 0.982769623484365, 0.9808551372048501, 0.98468410976388, 0.9814932992980216, 0.9725590299936184, 0.9776643267389917, 0.9795788130185067, 0.98468410976388, 0.98468410976388, 0.9897894065092534, 0.9878749202297383, 0.9897894065092534, 0.9897894065092534, 0.9910657306955967, 0.990427568602425, 0.98851308232291, 0.990427568602425, 0.9872367581365666, 0.9878749202297383, 0.9891512444160817, 0.9891512444160817, 0.9897894065092534, 0.9910657306955967, 0.9897894065092534, 0.990427568602425, 0.990427568602425, 0.9897894065092534, 0.990427568602425, 0.9897894065092534, 0.990427568602425, 0.9897894065092534, 0.9897894065092534, 0.9897894065092534, 0.9891512444160817, 0.9897894065092534, 0.9891512444160817, 0.9910657306955967, 0.9891512444160817, 0.990427568602425]
test_loss_list: [0.29795434820104616, 0.2545284815702815, 0.19106738575810225, 0.17300039234248046, 0.2784603256589676, 0.09760268792932929, 0.2458925157148695, 0.11523071681719502, 0.1089724511791933, 0.10214529508529992, 0.04046894534941221, 0.0586602729278954, 0.0520744658035478, 0.04844568014187662, 0.06042136234196844, 0.07167122805668089, 0.08389739419555776, 0.05625860034802938, 0.049073121997192784, 0.04563511356655556, 0.02943906867541064, 0.03392321034497938, 0.0318247030552403, 0.03525970262720021, 0.03496064679711708, 0.035481813927840354, 0.035041112577253404, 0.03306664044356118, 0.036733086281466445, 0.04622225004849107, 0.033058025518714985, 0.032245846670367444, 0.03264013959017816, 0.0324867430106827, 0.03365988649483723, 0.032440468800944346, 0.03191463659555777, 0.03296014047209981, 0.02995422201493799, 0.03572327769744538, 0.03311371788999056, 0.033105142529129064, 0.031857832900901224, 0.03387780729596615, 0.03562459421486057, 0.03311622010384169, 0.03408174898846037, 0.03160856055480081, 0.033353695002006134, 0.03458530518430646]
val_acc_list: [0.834077855775367, 0.9055520102105935, 0.9234205488194002, 0.9278876834716018, 0.8513082322910019, 0.9597957881301851, 0.9061901723037652, 0.9444798978940651, 0.9553286534779835, 0.9693682195277601, 0.98468410976388, 0.9814932992980216, 0.982769623484365, 0.9744735162731334, 0.9865985960433951, 0.9757498404594767, 0.9706445437141034, 0.978940650925335, 0.9763880025526483, 0.98468410976388, 0.9891512444160817, 0.9872367581365666, 0.98851308232291, 0.9891512444160817, 0.9859604339502234, 0.9872367581365666, 0.9878749202297383, 0.990427568602425, 0.9872367581365666, 0.9840459476707084, 0.9897894065092534, 0.9878749202297383, 0.9897894065092534, 0.9878749202297383, 0.9891512444160817, 0.98851308232291, 0.9878749202297383, 0.9891512444160817, 0.990427568602425, 0.98851308232291, 0.98851308232291, 0.9897894065092534, 0.98851308232291, 0.98851308232291, 0.9897894065092534, 0.98851308232291, 0.990427568602425, 0.990427568602425, 0.9910657306955967, 0.9910657306955967]
val_loss_list [0.4202929409791012, 0.26718779881389776, 0.2143127285514255, 0.20730370783950297, 0.42391405139493815, 0.10609758363285919, 0.2704242745186297, 0.16620249257717587, 0.11582804004761524, 0.09064789281739873, 0.04782469969656204, 0.055225069308534686, 0.054052445450702646, 0.07588324016429322, 0.048647273164979014, 0.06883494697192837, 0.09124579124939058, 0.053917782112708965, 0.07481177356860325, 0.041660662955183975, 0.036183014681636434, 0.0380767902177616, 0.03289527107803484, 0.03726475746552008, 0.04046395782390799, 0.03642424488661702, 0.032870251675729156, 0.03198373101741356, 0.03699041597354487, 0.04826259277377881, 0.03763201708805317, 0.03563894799210212, 0.033398811723600375, 0.035828615871688105, 0.0322237012628555, 0.03662988833390528, 0.0340519390400524, 0.0328588119207, 0.03205537675696215, 0.03600968108118223, 0.03698202120106042, 0.032306784831865915, 0.03276363726641017, 0.03374911105151559, 0.03374919960652751, 0.034900131804199784, 0.03511282258047259, 0.033842112137345066, 0.03263884022286517, 0.03749613561658593]
acc: 0.990
precision_1: 0.990
NAR_1: 0.003
F1_score_1: 0.994
precision_2: 0.988
NAR_2: 0.012
F1_score_2: 0.988
precision_3: 0.977
NAR_3: 0.004
F1_score_3: 0.986
precision_4: 0.996
NAR_4: 0.013
F1_score_4: 0.991
precision_5: 1.000
NAR_5: 0.004
F1_score_5: 0.998
precision_6: 0.992
NAR_6: 0.024
F1_score_6: 0.984
acc: 0.991
precision_1: 1.000
NAR_1: 0.003
F1_score_1: 0.998
precision_2: 0.988
NAR_2: 0.020
F1_score_2: 0.984
precision_3: 0.984
NAR_3: 0.000
F1_score_3: 0.992
precision_4: 0.987
NAR_4: 0.013
F1_score_4: 0.987
precision_5: 0.989
NAR_5: 0.000
F1_score_5: 0.995
precision_6: 0.996
NAR_6: 0.020
F1_score_6: 0.988

'''

'''
lr=3e-3 0.3
/home/ligong1/anaconda3/envs/airs/bin/python /home/ligong1/zqy_project/Phi-OTDR_dataset_and_code/ConvFormer_kqv.py 
epoch-max: 0
lr: 0.0009
Epoch 0 test_accuracy 0.77217613 test_loss 0.6210
Epoch 0 val_accuracy 0.73707722 val_loss 0.6701
epoch-max: 1
lr: 0.0009
Epoch 1 test_accuracy 0.90363752 test_loss 0.2775
Epoch 1 val_accuracy 0.88768347 val_loss 0.3058
epoch-max: 2
lr: 0.0009
Epoch 2 test_accuracy 0.80408424 test_loss 0.5005
Epoch 2 val_accuracy 0.82961072 val_loss 0.4450
epoch-max: 3
lr: 0.0009
Epoch 3 test_accuracy 0.87172942 test_loss 0.3330
Epoch 3 val_accuracy 0.89534142 val_loss 0.2881
epoch-max: 4
lr: 0.0009
Epoch 4 test_accuracy 0.85513720 test_loss 0.4243
Epoch 4 val_accuracy 0.87874920 val_loss 0.3224
epoch-max: 5
lr: 0.0009
Epoch 5 test_accuracy 0.91129547 test_loss 0.2664
Epoch 5 val_accuracy 0.88959796 val_loss 0.3104
epoch-max: 6
lr: 0.0009
Epoch 6 test_accuracy 0.79897894 test_loss 0.5970
Epoch 6 val_accuracy 0.84045948 val_loss 0.4939
epoch-max: 7
lr: 0.0009
Epoch 7 test_accuracy 0.93107849 test_loss 0.2205
Epoch 7 val_accuracy 0.94065093 val_loss 0.1708
epoch-max: 8
lr: 0.0009
Epoch 8 test_accuracy 0.87619655 test_loss 0.3485
Epoch 8 val_accuracy 0.87428207 val_loss 0.3639
epoch-max: 9
lr: 0.0009
Epoch 9 test_accuracy 0.73643906 test_loss 0.8257
Epoch 9 val_accuracy 0.80408424 val_loss 0.6386
epoch-max: 10
lr: 0.00027
Epoch 10 test_accuracy 0.98213146 test_loss 0.0446
Epoch 10 val_accuracy 0.97638800 val_loss 0.0716
epoch-max: 11
lr: 0.00027
Epoch 11 test_accuracy 0.98340779 test_loss 0.0484
Epoch 11 val_accuracy 0.97447352 val_loss 0.0645
epoch-max: 12
lr: 0.00027
Epoch 12 test_accuracy 0.99042757 test_loss 0.0304
Epoch 12 val_accuracy 0.98596043 val_loss 0.0393
epoch-max: 13
lr: 0.00027
Epoch 13 test_accuracy 0.98723676 test_loss 0.0390
Epoch 13 val_accuracy 0.98596043 val_loss 0.0518
epoch-max: 14
lr: 0.00027
Epoch 14 test_accuracy 0.98723676 test_loss 0.0476
Epoch 14 val_accuracy 0.98085514 val_loss 0.0563
epoch-max: 15
lr: 0.00027
Epoch 15 test_accuracy 0.97128271 test_loss 0.1046
Epoch 15 val_accuracy 0.97702616 val_loss 0.0698
epoch-max: 16
lr: 0.00027
Epoch 16 test_accuracy 0.98659860 test_loss 0.0354
Epoch 16 val_accuracy 0.97319719 val_loss 0.0743
epoch-max: 17
lr: 0.00027
Epoch 17 test_accuracy 0.98915124 test_loss 0.0406
Epoch 17 val_accuracy 0.97702616 val_loss 0.0827
epoch-max: 18
lr: 0.00027
Epoch 18 test_accuracy 0.96745373 test_loss 0.0923
Epoch 18 val_accuracy 0.95213784 val_loss 0.1619
epoch-max: 19
lr: 0.00027
Epoch 19 test_accuracy 0.95469049 test_loss 0.1669
Epoch 19 val_accuracy 0.96298660 val_loss 0.1246
epoch-max: 20
lr: 8.1e-05
Epoch 20 test_accuracy 0.98851308 test_loss 0.0331
Epoch 20 val_accuracy 0.98532227 val_loss 0.0499
epoch-max: 21
lr: 8.1e-05
Epoch 21 test_accuracy 0.98978941 test_loss 0.0250
Epoch 21 val_accuracy 0.98276962 val_loss 0.0486
epoch-max: 22
lr: 8.1e-05
Epoch 22 test_accuracy 0.99170389 test_loss 0.0299
Epoch 22 val_accuracy 0.98404595 val_loss 0.0498
epoch-max: 23
lr: 8.1e-05
Epoch 23 test_accuracy 0.99170389 test_loss 0.0233
Epoch 23 val_accuracy 0.98404595 val_loss 0.0466
epoch-max: 24
lr: 8.1e-05
Epoch 24 test_accuracy 0.99106573 test_loss 0.0263
Epoch 24 val_accuracy 0.98532227 val_loss 0.0411
epoch-max: 25
lr: 8.1e-05
Epoch 25 test_accuracy 0.98915124 test_loss 0.0274
Epoch 25 val_accuracy 0.98404595 val_loss 0.0534
epoch-max: 26
lr: 8.1e-05
Epoch 26 test_accuracy 0.98787492 test_loss 0.0376
Epoch 26 val_accuracy 0.97894065 val_loss 0.0711
epoch-max: 27
lr: 8.1e-05
Epoch 27 test_accuracy 0.98978941 test_loss 0.0293
Epoch 27 val_accuracy 0.98404595 val_loss 0.0517
epoch-max: 28
lr: 8.1e-05
Epoch 28 test_accuracy 0.99042757 test_loss 0.0263
Epoch 28 val_accuracy 0.98532227 val_loss 0.0503
epoch-max: 29
lr: 8.1e-05
Epoch 29 test_accuracy 0.99361838 test_loss 0.0250
Epoch 29 val_accuracy 0.98532227 val_loss 0.0476
epoch-max: 30
lr: 2.43e-05
Epoch 30 test_accuracy 0.99234205 test_loss 0.0236
Epoch 30 val_accuracy 0.98723676 val_loss 0.0474
epoch-max: 31
lr: 2.43e-05
Epoch 31 test_accuracy 0.99234205 test_loss 0.0257
Epoch 31 val_accuracy 0.98723676 val_loss 0.0458
epoch-max: 32
lr: 2.43e-05
Epoch 32 test_accuracy 0.99298022 test_loss 0.0248
Epoch 32 val_accuracy 0.98659860 val_loss 0.0518
epoch-max: 33
lr: 2.43e-05
Epoch 33 test_accuracy 0.99298022 test_loss 0.0225
Epoch 33 val_accuracy 0.98915124 val_loss 0.0428
epoch-max: 34
lr: 2.43e-05
Epoch 34 test_accuracy 0.99361838 test_loss 0.0216
Epoch 34 val_accuracy 0.98787492 val_loss 0.0436
epoch-max: 35
lr: 2.43e-05
Epoch 35 test_accuracy 0.99298022 test_loss 0.0223
Epoch 35 val_accuracy 0.98532227 val_loss 0.0460
epoch-max: 36
lr: 2.43e-05
Epoch 36 test_accuracy 0.99425654 test_loss 0.0206
Epoch 36 val_accuracy 0.98787492 val_loss 0.0465
epoch-max: 37
lr: 2.43e-05
Epoch 37 test_accuracy 0.99298022 test_loss 0.0238
Epoch 37 val_accuracy 0.98532227 val_loss 0.0415
epoch-max: 38
lr: 2.43e-05
Epoch 38 test_accuracy 0.99298022 test_loss 0.0232
Epoch 38 val_accuracy 0.98723676 val_loss 0.0472
epoch-max: 39
lr: 2.43e-05
Epoch 39 test_accuracy 0.99298022 test_loss 0.0224
Epoch 39 val_accuracy 0.98787492 val_loss 0.0431
epoch-max: 40
lr: 7.29e-06
Epoch 40 test_accuracy 0.99425654 test_loss 0.0228
Epoch 40 val_accuracy 0.98851308 val_loss 0.0457
epoch-max: 41
lr: 7.29e-06
Epoch 41 test_accuracy 0.99425654 test_loss 0.0228
Epoch 41 val_accuracy 0.98723676 val_loss 0.0429
epoch-max: 42
lr: 7.29e-06
Epoch 42 test_accuracy 0.99553287 test_loss 0.0199
Epoch 42 val_accuracy 0.98787492 val_loss 0.0473
epoch-max: 43
lr: 7.29e-06
Epoch 43 test_accuracy 0.99298022 test_loss 0.0234
Epoch 43 val_accuracy 0.98723676 val_loss 0.0480
epoch-max: 44
lr: 7.29e-06
Epoch 44 test_accuracy 0.99298022 test_loss 0.0265
Epoch 44 val_accuracy 0.98915124 val_loss 0.0434
epoch-max: 45
lr: 7.29e-06
Epoch 45 test_accuracy 0.99425654 test_loss 0.0254
Epoch 45 val_accuracy 0.98787492 val_loss 0.0440
epoch-max: 46
lr: 7.29e-06
Epoch 46 test_accuracy 0.99425654 test_loss 0.0204
Epoch 46 val_accuracy 0.98659860 val_loss 0.0464
epoch-max: 47
lr: 7.29e-06
Epoch 47 test_accuracy 0.99298022 test_loss 0.0237
Epoch 47 val_accuracy 0.98723676 val_loss 0.0470
epoch-max: 48
lr: 7.29e-06
Epoch 48 test_accuracy 0.99425654 test_loss 0.0208
Epoch 48 val_accuracy 0.98723676 val_loss 0.0447
epoch-max: 49
lr: 7.29e-06
Epoch 49 test_accuracy 0.99361838 test_loss 0.0210
Epoch 49 val_accuracy 0.98659860 val_loss 0.0435
bestvacc: 0.9891512444160817
besttest: 0.9929802169751116
train totally using %.3f seconds  22542.67163181305
train_acc_list: [0.7284019874979965, 0.8737778490142651, 0.905433563071005, 0.9213816316717423, 0.9329219426190094, 0.9420580221189293, 0.9523962173425229, 0.9510338195223593, 0.961211732649463, 0.9584869370091361, 0.9854143292194262, 0.9920660362237538, 0.9948709729123257, 0.9943099855746114, 0.9949511139605706, 0.9918256130790191, 0.9939092803333868, 0.9916653309825293, 0.9939894213816317, 0.9955121012982849, 0.9973553454079179, 0.999038307421061, 0.999118448469306, 0.9996794358070203, 0.9995191537105306, 0.9991985895175509, 0.9987177432280814, 0.999038307421061, 0.9989581663728162, 0.9994390126622856, 1.0, 1.0, 0.9999198589517551, 0.9998397179035102, 0.9998397179035102, 1.0, 0.9998397179035102, 0.9996794358070203, 1.0, 0.9998397179035102, 0.9998397179035102, 1.0, 0.9999198589517551, 1.0, 1.0, 1.0, 1.0, 0.9999198589517551, 1.0, 1.0]
train_loss_list: [0.7126038838321568, 0.35105909271151364, 0.2573642859326383, 0.2168477974487301, 0.1910512814786678, 0.16699522785162482, 0.13554879895634087, 0.13867429088188243, 0.11512388973939772, 0.11647695091440508, 0.04419094413593124, 0.024774601041688882, 0.017966216990522497, 0.016836145687890228, 0.016601168160454864, 0.024814568513182558, 0.018631117072529197, 0.02520848254288324, 0.01879004592722064, 0.01466128735962877, 0.007643044660027094, 0.003275820212838226, 0.0035013704226863423, 0.0027350936704508273, 0.0023584452675144336, 0.0033944901498850102, 0.004488545157092236, 0.0029567544453599757, 0.004016353303904673, 0.0024777596015429274, 0.0012209668147100399, 0.0008715212970762125, 0.0011861429707037798, 0.0009462063058039591, 0.0008964803217911577, 0.0007943838314896337, 0.0011314136403958298, 0.0012104346070188434, 0.0007676843736176113, 0.0008087012286226602, 0.0007231303335340744, 0.0004888010609745806, 0.0007513185725322174, 0.0004888714386378732, 0.0005060726203377374, 0.0005092166425327841, 0.0005154813256793761, 0.0007081499500958049, 0.0006139673522035283, 0.0005923199602003598]
test_acc_list: [0.7721761327377153, 0.9036375239310785, 0.8040842373962986, 0.8717294192724953, 0.8551372048500319, 0.9112954690491385, 0.7989789406509253, 0.9310784939374601, 0.8761965539246969, 0.7364390555201021, 0.9821314613911933, 0.9834077855775367, 0.990427568602425, 0.9872367581365666, 0.9872367581365666, 0.971282705807275, 0.9865985960433951, 0.9891512444160817, 0.967453733248245, 0.9546904913848118, 0.98851308232291, 0.9897894065092534, 0.9917038927887684, 0.9917038927887684, 0.9910657306955967, 0.9891512444160817, 0.9878749202297383, 0.9897894065092534, 0.990427568602425, 0.9936183790682833, 0.99234205488194, 0.99234205488194, 0.9929802169751116, 0.9929802169751116, 0.9936183790682833, 0.9929802169751116, 0.994256541161455, 0.9929802169751116, 0.9929802169751116, 0.9929802169751116, 0.994256541161455, 0.994256541161455, 0.9955328653477984, 0.9929802169751116, 0.9929802169751116, 0.994256541161455, 0.994256541161455, 0.9929802169751116, 0.994256541161455, 0.9936183790682833]
test_loss_list: [0.6210164196257081, 0.2774666576570242, 0.5005192264382329, 0.3329964455464209, 0.42431453976552097, 0.2663933400691924, 0.5969792703233127, 0.22045962555257945, 0.3484713289202476, 0.8257323874217667, 0.04459550531464629, 0.04837792198742712, 0.030432555035670758, 0.03903857968952234, 0.04760919614102245, 0.1045569954529091, 0.03537610513576881, 0.04064062679037529, 0.09230319807718435, 0.16693572356049044, 0.033107253452336796, 0.024956932607669933, 0.02990457563876645, 0.02325638757904425, 0.02632674006362118, 0.027375670034899133, 0.03757521582232287, 0.029288833165064344, 0.026285275444265885, 0.02497398455296912, 0.023583219532806003, 0.025742943749088337, 0.02479134403194287, 0.022509336628473296, 0.02156567696542601, 0.022281834389271966, 0.020562349522170197, 0.02383215693543188, 0.023198512960056067, 0.022422446600957095, 0.022788679862858364, 0.022768824481336568, 0.019854713307555654, 0.023383905032237726, 0.02649220642032727, 0.02538381829008915, 0.020417354076155844, 0.023673416713438422, 0.020755137355044265, 0.021012005155301137]
val_acc_list: [0.7370772176132737, 0.8876834716017868, 0.8296107211231653, 0.8953414167198468, 0.8787492022973835, 0.8895979578813018, 0.8404594767070837, 0.9406509253350351, 0.8742820676451819, 0.8040842373962986, 0.9763880025526483, 0.9744735162731334, 0.9859604339502234, 0.9859604339502234, 0.9808551372048501, 0.97702616464582, 0.97319719208679, 0.97702616464582, 0.9521378430121251, 0.9629865985960434, 0.9853222718570517, 0.982769623484365, 0.9840459476707084, 0.9840459476707084, 0.9853222718570517, 0.9840459476707084, 0.978940650925335, 0.9840459476707084, 0.9853222718570517, 0.9853222718570517, 0.9872367581365666, 0.9872367581365666, 0.9865985960433951, 0.9891512444160817, 0.9878749202297383, 0.9853222718570517, 0.9878749202297383, 0.9853222718570517, 0.9872367581365666, 0.9878749202297383, 0.98851308232291, 0.9872367581365666, 0.9878749202297383, 0.9872367581365666, 0.9891512444160817, 0.9878749202297383, 0.9865985960433951, 0.9872367581365666, 0.9872367581365666, 0.9865985960433951]
val_loss_list [0.670101669674017, 0.3058450932766558, 0.4449622078741692, 0.28806921413966585, 0.32237882883649094, 0.3103953211245184, 0.4939155310763008, 0.17082303032286142, 0.3638854787699232, 0.6385536632804695, 0.07157696084248862, 0.06448724841620602, 0.03933715640789383, 0.05176552986939692, 0.05629505150188509, 0.06979570982373992, 0.07432239950822823, 0.08274405258408497, 0.16194394340197263, 0.12462835666527247, 0.049889871363584644, 0.048576673887138745, 0.04984735853757192, 0.04655524750464514, 0.04107666519065584, 0.05336668668983015, 0.07111899323752145, 0.051739259010826905, 0.05028568819772995, 0.04759697731303592, 0.04735726362300089, 0.04579494546240832, 0.05183614665761803, 0.042785773509505666, 0.04360713147989558, 0.04596494153279031, 0.04653412171823294, 0.041530332370118175, 0.04717061592142198, 0.043069984485109854, 0.04572313608957988, 0.04293401246480485, 0.047347733682059065, 0.04797275238775465, 0.043390188521531245, 0.04400169048123495, 0.04641331495659703, 0.04697388126484783, 0.044704663605592276, 0.043544463929135545]
acc: 0.994
precision_1: 0.990
NAR_1: 0.003
F1_score_1: 0.994
precision_2: 0.996
NAR_2: 0.008
F1_score_2: 0.994
precision_3: 0.988
NAR_3: 0.004
F1_score_3: 0.992
precision_4: 1.000
NAR_4: 0.009
F1_score_4: 0.996
precision_5: 0.993
NAR_5: 0.004
F1_score_5: 0.995
precision_6: 0.996
NAR_6: 0.012
F1_score_6: 0.992
acc: 0.987
precision_1: 0.990
NAR_1: 0.003
F1_score_1: 0.994
precision_2: 0.976
NAR_2: 0.024
F1_score_2: 0.976
precision_3: 0.988
NAR_3: 0.008
F1_score_3: 0.990
precision_4: 0.983
NAR_4: 0.008
F1_score_4: 0.987
precision_5: 0.996
NAR_5: 0.011
F1_score_5: 0.993
precision_6: 0.983
NAR_6: 0.029
F1_score_6: 0.977

Process finished with exit code 0

'''
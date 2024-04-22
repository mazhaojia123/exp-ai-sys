import os
import scipy.io
import torch
import torch.nn as nn
from collections import OrderedDict

os.putenv('MLU_VISIBLE_DEVICES','')
cfgs = [64,'R', 64,'R', 'M', 128,'R', 128,'R', 'M',
       256,'R', 256,'R', 256,'R', 256,'R', 'M', 
       512,'R', 512,'R', 512,'R', 512,'R', 'M',
        512,'R', 512,'R', 512,'R', 512,'R', 'M']

IMAGE_PATH = 'data/strawberries.jpg'
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'

def vgg19():
    layers = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
        'flatten', 'fc6', 'relu6','fc7', 'relu7', 'fc8', 'softmax'
    ]
    layer_container = nn.Sequential()
    in_channels = 3
    num_classes = 1000
    for i, layer_name in enumerate(layers):
        if layer_name.startswith('conv'):
            # TODO: 在时序容器中传入卷积运算
            layer_container.append(nn.Conv2d(in_channels, cfgs[i], kernel_size=(3, 3), padding=(1, 1)))
            in_channels = cfgs[i]
        elif layer_name.startswith('relu'):
            # TODO: 在时序容器中执行ReLU计算
            layer_container.append(nn.ReLU())
        elif layer_name.startswith('pool'):
            # TODO: 在时序容器中执行maxpool计算
            layer_container.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif layer_name == 'flatten':
            # TODO: 在时序容器中执行flatten计算 
            layer_container.append(nn.Flatten())
        elif layer_name == 'fc6':
            # TODO: 在时序容器中执行全连接层计算
            layer_container.append(nn.Linear(512*7*7, 4096))
        elif layer_name == 'fc7':
            # TODO: 在时序容器中执行全连接层计算
            layer_container.append(nn.Linear(4096, 4096))
        elif layer_name == 'fc8':
            # TODO: 在时序容器中执行全连接层计算
            layer_container.append(nn.Linear(4096, num_classes))
        elif layer_name == 'softmax':
            # TODO: 在时序容器中执行Softmax计算
            layer_container.append(nn.Softmax(dim=1))
    return layer_container


if __name__ == '__main__':
    #TODO:使用scipy加载.mat格式的VGG19模型
    datas = scipy.io.loadmat(VGG_PATH)['layers'][0]

    model = vgg19()
    new_state_dict = OrderedDict()
    for i, param_name in enumerate(model.state_dict()):
        name = param_name.split('.')
        index = int(name[0])
        if index < 37:
            if name[-1] == 'weight':
                new_state_dict[param_name] = torch.from_numpy(datas[index][0][0][0][0][0]).float()
            else:
                new_state_dict[param_name] = torch.from_numpy(datas[index][0][0][0][0][1][0]).float()
        else:
            if name[-1] == 'weight':
                new_state_dict[param_name] = torch.from_numpy(datas[index-1][0][0][0][0][0]).float()
            else:
                new_state_dict[param_name] = torch.from_numpy(datas[index-1][0][0][0][0][1][0]).float()
    #TODO:加载网络参数到model
    for i, param_name in enumerate(new_state_dict):
        name = param_name.split('.')
        if name[-1] == 'weight':
            index = int(name[0])
            if index < 37:  # conv: []
                new_state_dict[param_name] = new_state_dict[param_name].permute(3, 2, 0, 1)
            else:           # fc
                sz = new_state_dict[param_name].size()
                shp = [sz[0]*sz[1]*sz[2], sz[3]]
                tmp = torch.reshape(new_state_dict[param_name], (shp[0], shp[1]))
                new_state_dict[param_name] = tmp.permute(1, 0)

    model.load_state_dict(new_state_dict)
    print("*** Start Saving pth ***")
    #TODO:保存模型的参数到models/vgg19.pth
    torch.save(model.state_dict(), 'models/vgg19.pth')
    print('Saving pth  PASS.')
    

import timm
import torch
import torch.nn as nn
import torchvision


'''
CNN ResNet based models
'''

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(ResNetBlock, self).__init__()
        self.num_layers = 18
        
        self.convolution1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        self.batchnorm1 = nn.BatchNorm2d(out_channels)

        self.convloution2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.batchnorm2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.convolution1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.convloution2(x)
        x = self.batchnorm2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResNet18Model(nn.Module):
    def __init__(self, block, num_classes=7):
        super(ResNet18Model, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self.make_layers(block, intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(block, intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(block, intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(block, intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_phase = nn.Linear(512, num_classes)



    def forward(self, x):


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)


        out = self.fc_phase(x)
        
        return out

    def make_layers(self, block, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels))
        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels
        layers.append(block(self.in_channels, intermediate_channels))
        return nn.Sequential(*layers)
    

class ResNet18LSTM(nn.Module):
    def __init__(self, num_classes=7, lstm_hidden_n = 512):
        super(ResNet18LSTM, self).__init__()
        self.model = ResNet18Model(ResNetBlock)
        model_path = 'models/resnet_18_ord_model.pth'
        self.model.load_state_dict(torch.load(model_path))

        self.num_ftrs = self.model.fc_phase.in_features

        self.model.fc_phase = nn.Linear(self.num_ftrs ,self.num_ftrs)

        nn.init.xavier_uniform_(self.model.fc_phase.weight)


        # self.in_channels = 64
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        # self.layer1 = self.make_layers(block, intermediate_channels=64, stride=1)
        # self.layer2 = self.make_layers(block, intermediate_channels=128, stride=2)
        # self.layer3 = self.make_layers(block, intermediate_channels=256, stride=2)
        # self.layer4 = self.make_layers(block, intermediate_channels=512, stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc_tool = nn.Linear(512, num_classes)

        self.linear_hidden = nn.Linear(lstm_hidden_n,lstm_hidden_n)
        self.linear_phase = nn.Linear(lstm_hidden_n,num_classes)
        self.lstm = nn.LSTM(lstm_hidden_n, lstm_hidden_n, batch_first=True)


        self.drop = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.batchnorm1d = nn.BatchNorm1d(num_features=lstm_hidden_n)


    def forward(self, x):


        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.reshape(x.shape[0], -1)

        # #tool out
        # out_tool = self.fc_tool(x)
        # out_tool = self.sigmoid(out_tool)
        x = self.model(x)

        #lstm process
        self.lstm.flatten_parameters()
        out_phase,_ = self.lstm(x)

        #linear out
        out_phase = self.linear_hidden(out_phase)
        out_phase = self.relu(out_phase)
        out_phase = self.batchnorm1d(out_phase)
        out_phase = self.drop(out_phase)

        out_phase = self.linear_phase(out_phase)


        return  out_phase

    # def make_layers(self, block, intermediate_channels, stride):
    #     layers = []

    #     identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels, kernel_size=1, stride=stride),
    #                                         nn.BatchNorm2d(intermediate_channels))
    #     layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
    #     self.in_channels = intermediate_channels
    #     layers.append(block(self.in_channels, intermediate_channels))
    #     return nn.Sequential(*layers)


class ResNet50(nn.Module):
    def __init__(self,num_classes=7, num_tools=7):
        super(ResNet50, self).__init__()
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(self.num_ftrs ,num_classes)

        nn.init.xavier_uniform_(self.model.fc.weight)


    def forward(self, x):
        out = self.model(x)
    
        return out
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

class ResNet50_ST(nn.Module):
    def __init__(self,num_classes=7, num_tools=7):
        super(ResNet50_ST, self).__init__()
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(self.num_ftrs ,num_classes)
        self.sig = nn.Sigmoid() 
        nn.init.xavier_uniform_(self.model.fc.weight)


    def forward(self, x):
        out = self.model(x)
    
        return out,self.sig(out)

class ResNet50_ST_Phase(nn.Module):
    def __init__(self,num_classes=7, num_tools=7):
        super(ResNet50_ST_Phase, self).__init__()
        self.model = ResNet50_ST()
        model_path = 'models/ResNet50_od_model.pth'
        self.model.load_state_dict(torch.load(model_path))

        self.num_ftrs = self.model.model.fc.in_features

        self.model.model.fc = nn.Linear(self.num_ftrs ,num_classes)

        nn.init.xavier_uniform_(self.model.model.fc.weight)


    def forward(self, x):
        out = self.model(x)
    
        return out
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.model.fc.parameters():
            param.requires_grad = True

    
    # def freeze(self):
    #     for param in self.model.parameters():
    #         param.requires_grad = False

    #     for param in self.model.fc.parameters():
    #         param.requires_grad = True

class Xception(nn.Module):
    def __init__(self,num_classes=7):
        super(Xception, self).__init__()
        self.model = timm.create_model('xception', pretrained=True)
        self.num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(self.num_ftrs ,num_classes)

        nn.init.xavier_uniform_(self.model.fc.weight)


    def forward(self, x):
        out = self.model(x)
    
        return out
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True


class ResNet50LSTM(nn.Module):
    def __init__(self, num_classes=7, lstm_hidden_n = 512):
        super(ResNet50LSTM, self).__init__()
        self.model = ResNet50()
        model_path = 'models/resnet_50_ord_model.pth'
        self.model.load_state_dict(torch.load(model_path))

        self.num_ftrs = self.model.model.fc.in_features

        self.model.model.fc = nn.Linear(self.num_ftrs ,lstm_hidden_n)

        self.linear_hidden = nn.Linear(lstm_hidden_n,lstm_hidden_n)
        self.linear_phase = nn.Linear(lstm_hidden_n,num_classes)
        self.lstm = nn.LSTM(lstm_hidden_n, lstm_hidden_n, batch_first=True)


        self.drop = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.batchnorm1d = nn.BatchNorm1d(num_features=lstm_hidden_n)


        nn.init.xavier_uniform_(self.linear_phase.weight)

        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"): nn.init.orthogonal_(param) 


    def forward(self, x):

        x = self.model(x)



        #lstm process
        self.lstm.flatten_parameters()
        out_phase,_ = self.lstm(x)

        #linear out
        # out_phase = self.linear_hidden(out_phase)
        # out_phase = self.relu(out_phase)
        # out_phase = self.batchnorm1d(out_phase)
        # out_phase = self.drop(out_phase)

        out_phase = self.linear_phase(out_phase)


        return  out_phase


    # def freeze(self):
    #     # To freeze the residual layers
    #     for param in self.model.parameters():
    #         param.requires_grad = False

    #     # for param in self.model.fc.parameters():
    #     #     param.requires_grad = True




'''
Transformer Models
'''
class ViT(nn.Module):
    def __init__(self, num_classes=7):
        super(ViT,self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.head.parameters():
            param.requires_grad = True

class ViT_LSTM(nn.Module):
    def __init__(self, num_classes=7):
        super(ViT_LSTM,self).__init__()
        self.model = ViT()
        model_path = 'models/ViT_ord_model.pth'
        self.model.load_state_dict(torch.load(model_path))
        self.num_ftrs = self.model.model.head.in_features
        self.model.model.head = nn.Linear(self.num_ftrs ,self.num_ftrs)

        self.linear_hidden = nn.Linear(self.num_ftrs,self.num_ftrs)
        self.linear_phase = nn.Linear(self.num_ftrs,num_classes)
        self.lstm = nn.LSTM(self.num_ftrs, self.num_ftrs, batch_first=True)


        self.drop = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.batchnorm1d = nn.BatchNorm1d(num_features=self.num_ftrs)

        nn.init.xavier_uniform_(self.linear_phase.weight)
        nn.init.xavier_uniform_(self.linear_hidden.weight)
        nn.init.xavier_uniform_(self.model.model.head.weight)

        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"): nn.init.orthogonal_(param) 
        self.model.model.head

        

    def forward(self, x):
        x = self.model(x)

        #lstm process
        self.lstm.flatten_parameters()
        out_phase,_ = self.lstm(x)

        #linear out
        # out_phase = self.linear_hidden(out_phase)
        # out_phase = self.relu(out_phase)
        # out_phase = self.batchnorm1d(out_phase)
        # out_phase = self.drop(out_phase)

        out_phase = self.linear_phase(out_phase)


        return  out_phase
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.model.head.parameters():
            param.requires_grad = True




class ViT_OD(nn.Module):
    def __init__(self, num_classes=7):
        super(ViT_OD,self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, num_classes)
        self.sig = nn.Sigmoid() 
        nn.init.xavier_uniform_(self.model.head.weight)

    def forward(self, x):
        x = self.model(x)
        return x,self.sig(x)
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.head.parameters():
            param.requires_grad = True


class ViT_OD_phase(nn.Module):
    def __init__(self, num_classes=7):
        super(ViT_OD_phase,self).__init__()
        self.model = ViT_OD()
        model_path = 'models/ViT_ord_od_model.pth'
        self.model.load_state_dict(torch.load(model_path))


        n_features = self.model.model.head.in_features
        self.model.model.head = nn.Linear(n_features, num_classes)
        self.sig = nn.Sigmoid() 
        nn.init.xavier_uniform_(self.model.model.head.weight)

    def forward(self, x):
        x = self.model(x)
        return x,self.sig(x)
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.model.head.parameters():
            param.requires_grad = True

class DeiT(nn.Module):
    def __init__(self, num_classes=7):
        super(DeiT,self).__init__()
        self.model = timm.create_model('deit3_large_patch16_224_in21ft1k', pretrained=True)
        self.n_features = self.model.head.in_features
        self.lstm = nn.LSTM(self.n_features, self.n_features, batch_first=True)
        self.model.head = nn.Linear(self.n_features, self.n_features)
        self.linear_phase = nn.Linear(self.n_features,num_classes)

        nn.init.xavier_uniform_(self.linear_phase.weight)

        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"): nn.init.orthogonal_(param) 


    def forward(self, x):
        x = self.model(x)

        self.lstm.flatten_parameters()
        out_phase,_ = self.lstm(x)
        out_phase = self.linear_phase(out_phase)


        return  out_phase
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.head.parameters():
            param.requires_grad = True

class DeiT_Distilled(nn.Module):
    def __init__(self, num_classes=7):
        super(DeiT_Distilled, self).__init__()
        self.model = timm.create_model('deit_base_distilled_patch16_224', pretrained=True)
        n_features_head = self.model.head.in_features
        n_features_dist = self.model.head_dist.in_features
        self.model.head = nn.Linear(n_features_head, num_classes)
        self.model.head_dist = nn.Linear(n_features_dist, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.blocks[6:].parameters():
            param.requires_grad = True
        #Set grad true for both the head and the distill head
        for param in self.model.head.parameters():
            param.requires_grad = True
        for param in self.model.head_dist.parameters():
            param.requires_grad = True

class DeiT_Distilled_OD(nn.Module):
    def __init__(self, num_classes=7):
        super(DeiT_Distilled_OD,self).__init__()
        self.model = timm.create_model('deit_base_distilled_patch16_224', pretrained=True)
        n_features_head = self.model.head.in_features
        n_features_dist = self.model.head_dist.in_features
        self.model.head = nn.Linear(n_features_head, num_classes)
        self.model.head_dist = nn.Linear(n_features_dist, num_classes)
        self.sig = nn.Sigmoid() 
        nn.init.xavier_uniform_(self.model.head.weight)
        nn.init.xavier_uniform_(self.model.head_dist.weight)

    def forward(self, x):
        x = self.model(x)
        return x,self.sig(x)
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.blocks[8:].parameters():
            param.requires_grad = True
        #Set grad true for both the head and the distill head
        for param in self.model.head.parameters():
            param.requires_grad = True
        for param in self.model.head_dist.parameters():
            param.requires_grad = True


class DeiT_Distilled_OD_LSTM(nn.Module):
    def __init__(self, num_classes=7):
        super(DeiT_Distilled_OD_LSTM,self).__init__()

        self.model = DeiT_Distilled_OD()
        model_path = 'models/DeiT_ord_od_model.pth'
        self.model.load_state_dict(torch.load(model_path))


        n_features_head = self.model.model.head.in_features
        n_features_dist = self.model.model.head_dist.in_features
        self.model.model.head = nn.Linear(n_features_head, n_features_head)
        self.model.model.head_dist = nn.Linear(n_features_dist, n_features_dist)
        self.lstm = nn.LSTM(n_features_dist, n_features_dist, batch_first=True)
        self.linear_phase = nn.Linear(n_features_head,num_classes)

        nn.init.xavier_uniform_(self.model.model.head.weight)
        nn.init.xavier_uniform_(self.model.model.head_dist.weight)
        nn.init.xavier_uniform_(self.linear_phase.weight)

        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"): nn.init.orthogonal_(param) 

    def forward(self, x):
        x,_ = self.model(x)

        self.lstm.flatten_parameters()
        out_phase,_ = self.lstm(x)
        out_phase = self.linear_phase(out_phase)


        return out_phase
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.model.parameters():
            param.requires_grad = False

        #Set grad true for both the head and the distill head
        for param in self.model.model.head.parameters():
            param.requires_grad = True
        for param in self.model.model.head_dist.parameters():
            param.requires_grad = True

class DeiT_Distilled_LSTM(nn.Module):
    def __init__(self, num_classes=7,dataset=0):
        super(DeiT_Distilled_LSTM, self).__init__()
        self.model = DeiT_Distilled()
        # model_path = 'models/Ordered_DeiT_model_best.pth'
        if dataset == 0 :
            model_path = 'models/DeiT_ord_final_model.pth'
        else:
            model_path = 'models/DeiT_ord_32_model.pth'
            
        self.model.load_state_dict(torch.load(model_path))


        # self.model = timm.create_model('deit_base_distilled_patch16_224', pretrained=True)
        # n_features_head = self.model.head.in_features
        self.n_features = self.model.model.head_dist.in_features
        self.lstm = nn.LSTM(self.n_features, self.n_features, batch_first=True)
        self.model.model.head = nn.Linear(self.n_features, self.n_features)
        self.model.model.head_dist = nn.Linear(self.n_features, self.n_features)
        self.linear_phase = nn.Linear(self.n_features,num_classes)

        nn.init.xavier_uniform_(self.linear_phase.weight)

        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"): nn.init.orthogonal_(param) 


    def forward(self, x):
        x = self.model(x)

        self.lstm.flatten_parameters()
        out_phase,_ = self.lstm(x)
        out_phase = self.linear_phase(out_phase)


        return  out_phase
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.model.parameters():
            param.requires_grad = False

        #Set grad true for both the head and the distill head
        for param in self.model.model.head.parameters():
            param.requires_grad = True
        for param in self.model.model.head_dist.parameters():
            param.requires_grad = True


# class DeiT_Distilled_LSTM(nn.Module):
#     def __init__(self, num_classes=7,lstm_features=512):
#         super(DeiT_Distilled_LSTM, self).__init__()
#         # self.model = DeiT_Distilled()
#         # # model_path = 'models/Ordered_DeiT_model_best.pth'
#         # model_path = 'models/DeiT_ord_final_model.pth'
#         # self.model.load_state_dict(torch.load(model_path))


#         self.model = timm.create_model('deit_base_distilled_patch16_224', pretrained=True)
#         # n_features_head = self.model.head.in_features
#         self.n_features = self.model.head_dist.in_features
#         self.lstm = nn.LSTM(self.n_features, self.n_features,num_layers=2, batch_first=True)
#         self.model.head = nn.Linear(self.n_features, self.n_features)
#         self.model.head_dist = nn.Linear(self.n_features, self.n_features)
#         self.linear_phase = nn.Linear(self.n_features,num_classes)

#         nn.init.xavier_uniform_(self.linear_phase.weight)

#         for name, param in self.lstm.named_parameters():
#             if name.startswith("weight"): nn.init.orthogonal_(param) 


#     def forward(self, x):
#         x = self.model(x)

#         self.lstm.flatten_parameters()
#         out_phase,_ = self.lstm(x)
#         out_phase = self.linear_phase(out_phase)


#         return  out_phase
    
#     def freeze(self):
#         # To freeze the residual layers
#         for param in self.model.parameters():
#             param.requires_grad = False
#         # for param in self.model.blocks[8:].parameters():
#         #     param.requires_grad = True

#         #Set grad true for both the head and the distill head
#         for param in self.model.head.parameters():
#             param.requires_grad = True
#         for param in self.model.head_dist.parameters():
#             param.requires_grad = True


class Cross_Vit(nn.Module):
    def __init__(self, num_classes=7):
        super(Cross_Vit, self).__init__()
        self.model = timm.create_model('crossvit_18_dagger_240', pretrained=True)
        n_features_head0 = self.model.head[0].in_features
        n_features_head1 = self.model.head[1].in_features
        self.model.head[0] = nn.Linear(n_features_head0, num_classes)
        self.model.head[1] = nn.Linear(n_features_head1, num_classes)


    def forward(self, x):
        x = self.model(x)
        return x
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.model.parameters():
            param.requires_grad = False

        #Set grad true for both the head and the distill head
        for param in self.model.head.parameters():
            param.requires_grad = True
   
class BEIT(nn.Module):
    def __init__(self, num_classes=7):
        super(BEIT, self).__init__()
        self.model = timm.create_model('beit_base_patch16_224',pretrained=True)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.head.parameters():
            param.requires_grad = True

class CoAt(nn.Module):
    def __init__(self,num_classes=7):
        super(CoAt, self).__init__()
        self.model = timm.create_model('coatnet_2_rw_224',pretrained=True)
        self.num_ftrs = self.model.head.fc.in_features

        self.model.head.fc = nn.Linear(self.num_ftrs ,num_classes)

        nn.init.xavier_uniform_(self.model.head.fc.weight)


    def forward(self, x):
        out = self.model(x)
    
        return out
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.head.fc.parameters():
            param.requires_grad = True

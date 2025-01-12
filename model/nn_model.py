import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, resnet18, resnet50

def init_params(models):
    if not isinstance(models, list):
        models = [models]

    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) \
                    or isinstance(m, nn.Linear) \
                    or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.normal_(m.bias) if (m.bias is not None) else None
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, num_feature_map=64):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, 1, padding='same')
        self.conv1_bn=nn.BatchNorm2d(c_out)
        self.conv11 = nn.Conv2d(c_out, num_feature_map, 3, 1, padding='same')
        self.conv11_bn=nn.BatchNorm2d(num_feature_map)
        self.conv12 = nn.Conv2d(num_feature_map, c_out, 3, 1, padding='same')
        self.conv12_bn=nn.BatchNorm2d(c_out)


    def forward(self, x):
        x = self.conv1_bn(self.conv1(x))
        x = nn.ReLU()(x)
        xi = x
        x = self.conv11_bn(self.conv11(x))
        x = nn.ReLU()(x)
        x = self.conv12_bn(self.conv12(x))
        x = x + xi
        return x

    

class VAEClassifier(nn.Module):
    def __init__(self, num_feature_map=64):
        super(VAEClassifier, self).__init__()
        self.encoder_module = nn.ModuleList([BasicBlock(1, num_feature_map), 
                                             BasicBlock(num_feature_map, num_feature_map), 
                                             BasicBlock(num_feature_map, num_feature_map),
                                             BasicBlock(num_feature_map, num_feature_map)])
        
        self.decoder_module = nn.ModuleList([BasicBlock(num_feature_map, num_feature_map), 
                                             BasicBlock(num_feature_map, num_feature_map), 
                                             BasicBlock(num_feature_map, num_feature_map),
                                             BasicBlock(num_feature_map, num_feature_map)])
        
        self.output_layer = nn.ModuleList([BasicBlock(num_feature_map, 8), BasicBlock(8, 1)]) # decoder after upsampling
        self.output_layer2 = nn.Linear(num_feature_map, 10) # classification head
        self.mean_val = nn.Linear(num_feature_map, num_feature_map)
        self.var_val = nn.Linear(num_feature_map, num_feature_map)
        self.latent_dim = num_feature_map
        
    def forward(self, x, deterministic=True, classification_only=True):
        # encoding
        for i_op in self.encoder_module:
            x = i_op(x)
            x = nn.MaxPool2d(2, stride=2)(x)
        
        # encoder latent vector     
        mu = self.mean_val(torch.mean(x, dim=(2, 3)))
        logvar = self.var_val(torch.mean(x, dim=(2, 3)))
        
        if deterministic:
            sample_z = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            sample_z = eps * std + mu
        
        x = sample_z[:, :, None, None]
        z = x
        y = self.output_layer2(torch.mean(z, dim=(2, 3)))
        
        # decoding
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        for i_op in self.decoder_module:
            x = i_op(x)
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        
        for i_op in self.output_layer:
            x = i_op(x)
        
        # output
        x = x[:,:,2:30,2:30]
        x = torch.sigmoid(x) 
        
        if classification_only:
            return y
        else:
            return x, z, y, mu, logvar

    
class StAEClassifier(nn.Module):
    def __init__(self, num_feature_map=64):
        super(StAEClassifier, self).__init__()
        self.encoder_module = nn.ModuleList([BasicBlock(1, num_feature_map), 
                                             BasicBlock(num_feature_map, num_feature_map), 
                                             BasicBlock(num_feature_map, num_feature_map),
                                             BasicBlock(num_feature_map, num_feature_map)])
        
        self.decoder_module = nn.ModuleList([BasicBlock(num_feature_map, num_feature_map), 
                                             BasicBlock(num_feature_map, num_feature_map), 
                                             BasicBlock(num_feature_map, num_feature_map),
                                             BasicBlock(num_feature_map, num_feature_map)])
        
        self.output_layer = nn.ModuleList([BasicBlock(num_feature_map, 8), BasicBlock(8, 1)]) # decoder after upsampling
        self.output_layer2 = nn.Linear(num_feature_map, 10)
        
    def forward(self, x, classification_only=True):
        for i_op in self.encoder_module:
            x = i_op(x)
            x = nn.MaxPool2d(2, stride=2)(x)
        
        # encoder latent weight
        z = x
        y = self.output_layer2(torch.mean(z, dim=(2, 3)))
        
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        for i_op in self.decoder_module:
            x = i_op(x)
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        
        for i_op in self.output_layer:
            x = i_op(x)
        
        x = x[:,:,2:30,2:30]
        x = torch.sigmoid(x) 
        
        if classification_only:
            return y
        else:
            return x, z, y


class ResNetEnc(nn.Module):
    def __init__(self, image_size=32, **kwargs):
        super(ResNetEnc, self).__init__()
        self.image_size = image_size
        self.model = resnet50(pretrained=True)

        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        if self.image_size == 32:
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            delattr(self.model, 'maxpool')
            init_params(self.model.conv1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        # if hasattr(self.model, 'maxpool'): x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

        
class ResNetVAE(nn.Module):
    def __init__(self, REncoder, latent_dim=256):
        super(ResNetVAE, self).__init__()
        self.encoder_module = nn.ModuleList([REncoder])
        self.decoder_module = nn.ModuleList([BasicBlock(latent_dim, latent_dim), 
                                             BasicBlock(latent_dim, latent_dim), 
                                             BasicBlock(latent_dim, latent_dim),
                                             BasicBlock(latent_dim, latent_dim)])
        
        self.output_layer = nn.ModuleList([BasicBlock(latent_dim, 8), BasicBlock(8, 3)])
        self.output_layer2 = nn.Linear(latent_dim, 10)
        self.mean_val = nn.Linear(latent_dim * 8, latent_dim)
        self.var_val = nn.Linear(latent_dim * 8, latent_dim)
        self.latent_dim = latent_dim
        
    def forward(self, x, deterministic=True, classification_only=True):
        for i_op in self.encoder_module:
            x = i_op(x)
            x = nn.ReLU()(x)
        
        # encoder latent weight     
        mu = self.mean_val(x)
        logvar = self.var_val(x)
        
        if deterministic:
            sample_z = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            sample_z = eps * std + mu
        
        x = sample_z[:,:,None,None]
        z = x
        
        y = self.output_layer2(torch.mean(z, dim=(2, 3)))
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        for i_op in self.decoder_module:
            x = i_op(x)
            # x = nn.ReLU()(x)
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        
        for i_op in self.output_layer:
            # x = nn.ReLU()(x)
            x = i_op(x)
        
        x = torch.tanh(x) 
        if classification_only:
            return y
        else:
            return x, z, y, mu, logvar
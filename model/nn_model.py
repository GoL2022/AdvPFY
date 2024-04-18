import torch
import torch.nn as nn
import torch.nn.functional as F


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
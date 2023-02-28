# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import os
import numpy as np
from logger import Logger, SimpleLogger

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"]='2'
class MLP(nn.Module):   # define the MLP model
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_norm=True):
        '''
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: dimensionality of the output layer
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            batch_norm: batch_norm = True if use the normalize layer
        '''
        super(MLP, self).__init__()

        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.layers = nn.ModuleList()

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Multi-layer model
            self.norm_layers = nn.ModuleList()  # The normalize layers

            # Add the first layer
            self.layers.append(nn.Linear(input_dim, hidden_dim))

            # Add the hidden layers
            for layer in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

            # Add the last layer
            self.layers.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.norm_layers.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        # If use the normalize layer, then a linear layer is followed by a normalize layer.
        for layer in range(self.num_layers - 1):
            if self.batch_norm:
                x = F.relu(self.norm_layers[layer](self.layers[layer](x)))
            else:
                x = F.relu(self.layers[layer](x))
        return self.layers[self.num_layers - 1](x)


class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, enc_layer, dec_layer, dis_layer, batch_norm=True):
        '''
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: dimensionality of the output layer
            enc_layer: number of encoder layers in the neural networks
            dec_layer: number of decoder layers in the neural networks
            dis_layer: number of discriminator layers in the GAN (if use the MLP as the decoder)
            batch_norm: batch_norm = True if use the normalize layer
        '''
        super().__init__()
        self.encoder_c = MLP(input_dim, hidden_dim, output_dim, enc_layer, batch_norm)  # Define the E^C
        self.encoder_s = MLP(input_dim, hidden_dim, output_dim, enc_layer, batch_norm)  # Define the E^S
        self.decoder = MLP(2*output_dim, hidden_dim, input_dim, dec_layer, batch_norm)  # Define the D(C, S)

        self.discriminator = MLP(input_dim, hidden_dim, 1, dis_layer, batch_norm)

    def forward(self, x):
        X_c = self.encoder_c(x)  # C
        X_s = self.encoder_s(x)  # S
        X_rec = self.decoder(torch.concat([X_c, X_s], dim=1))   # D(C, S)

        s = torch.randn_like(X_s).to(X_s.device)   # Sample s form the standard Guassian distribution, and the dimension of s is same as X_s
        dec_noise = self.decoder(torch.concat([X_c, s], dim=1))  # X' = D(X_c, s)
        X_c_rec = self.encoder_c(dec_noise)  # C'
        X_s_rec = self.encoder_s(dec_noise)  # S'

        loss_rec_x = F.mse_loss(X_rec, x)   # L^X_recon
        loss_rec_c = F.mse_loss(X_c_rec, X_c)   # L^C_recon
        loss_rec_s = F.mse_loss(X_s_rec, s)  # L^S_recon
        
        # The GAN part
        # The positive sample, Discriminator(X)
        out_pos = self.discriminator(x)
        out_neg = self.discriminator(dec_noise.detach())
        ones = torch.ones_like(out_pos).to(out_pos.device)  # Label 1
        zeros = torch.zeros_like(out_pos).to(out_neg.device)    # Label 0

        out_discriminator = torch.concat([out_pos, out_neg])
        labels = torch.concat([ones, zeros])
        loss_discriminator = F.binary_cross_entropy_with_logits(out_discriminator, labels)

        # The generator part
        out = self.discriminator(dec_noise)   # Discriminator(X') = Discriminator(D(X_c, s)))
        loss_generator = F.binary_cross_entropy_with_logits(out, ones)

        return dec_noise, loss_generator, loss_rec_x, loss_rec_c, loss_rec_s, loss_discriminator


    
def transleted_X(feature_matrix):
    x = torch.tensor(feature_matrix).cuda()
    model = MyModel(feature_matrix.shape[1], 128, 64, 3, 3, 3, batch_norm=False).cuda()  # Define the model
    p_x =1
    p_c =1
    p_s =1
    p =1

    optim_generator = torch.optim.Adam([{
        'params': model.encoder_c.parameters(),
        'params': model.encoder_s.parameters(),
        'params': model.decoder.parameters(),
    }], lr=1e-3)
    optim_discriminator = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)
    
    
    # Train
    for epoch in range(500):
        optim_generator.zero_grad()
        optim_discriminator.zero_grad()

        dec_noise, loss_generator, loss_rec_x, loss_rec_c, loss_rec_s, loss_discriminator = model(x)
        loss = p*loss_generator + p_x*loss_rec_x + p_c*loss_rec_c + p_s* loss_rec_s

        loss.backward()
        loss_discriminator.backward()

        optim_generator.step()
        optim_discriminator.step()
        if epoch % 100 ==0:
            print(f'Epoch {epoch+1}/100: Loss total: {loss}, Loss discriminator: {loss_discriminator}')

    dec_noise, loss_generator, loss_rec_x, loss_rec_c, loss_rec_s, loss_discriminator = model(x)
    return dec_noise
    
    

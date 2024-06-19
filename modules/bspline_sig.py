import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bsplines(nn.Module):
    def __init__(self,
                    in_features,
                    out_features,
                    bias=True,
                    is_first=False,
                    omega0=-0.2, # a = 1.0
                    sigma0=15.0, # k = 10.0
                    trainable=False):
            super().__init__()
            self.omega_0 = omega0
            self.scale_0 = sigma0
            self.is_first = is_first

            self.in_features = in_features

            self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
            self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

            self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias)

    def forward(self, input):
        lin = self.linear(input)
        scale_in = self.scale_0 * lin
        coeff = self.omega_0
        is_neg = input[:,:,0]<0
        for i in range(is_neg.shape[1]):
            if is_neg[:,i].item():
                return 1/(1 + torch.exp(-scale_in + self.scale_0*coeff))
            else:
                return 1/(1 + torch.exp(scale_in + self.scale_0*coeff))

        
class INR(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 outermost_linear=True,
                 first_omega_0=-0.2,
                 hidden_omega_0=-0.2,
                 scale=15.0,
                 pos_encode=False,
                 sidelength=512,
                 fn_samples=None,
                 use_nyquist=True):
        super().__init__()

        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = Bsplines

        # Since complex numbers are two real numbers, reduce the number of
        # hidden parameters by 2
        #hidden_features = int(hidden_features / np.sqrt(2))
        #dtype = torch.cfloat
        self.complex = False
        #self.wavelet = 'gabor'

        # Legacy parameter
        self.pos_encode = False

        self.net = []
        self.net.append(
            self.nonlin(in_features,
                        hidden_features,
                        omega0=first_omega_0,
                        sigma0=scale,
                        is_first=True,
                        trainable=False))

        for i in range(hidden_layers):
            self.net.append(
                self.nonlin(hidden_features,
                            hidden_features,
                            omega0=hidden_omega_0,
                            sigma0=scale))
        
        if outermost_linear:
            if self.complex:
                dtype = torch.cfloat
            else:
                dtype = torch.float

            final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)
            self.net.append(final_linear)
        else:
            self.net.append(
                self.nonlin(hidden_features,
                            out_features,
                            omega0=hidden_omega_0,
                            sigma0=scale))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)

        #if self.wavelet == 'gabor':
         #   return output.real

        return output




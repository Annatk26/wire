import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
# from utils import quadratic_relu


class Bsplines_form(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            is_first=False,
            omega0=-0.2,  # a = 1.0
            sigma0=6.0,  # k = 10.0
            trainable=True):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        # self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def quadratic_relu(self, x):
        return torch.nn.ReLU()(x)**2

    def forward(self, input):
        lin = self.scale_0*self.linear(input)
        return (
            0.5 * self.quadratic_relu(lin + 1.5)
            - 1.5 * self.quadratic_relu(lin + 0.5)
            + 1.5 * self.quadratic_relu(lin - 0.5)
            - 0.5 * self.quadratic_relu(lin - 1.5)
        )
        # x_coord = lin[0,0]
        # relu = torch.nn.ReLU()
        # if -1.5 <= x_coord < -0.5:
        #     return 0.5 * relu(lin + 1.5)**2
        # elif -0.5 <= x_coord < 0.5:
        #     return -relu(lin + 1.5)**2 + 3 * relu(lin + 1.5) - 1.5
        #     # return 0.5*(-3 + 6*(relu(x-1))-2*relu(x-1)**2)
        # elif 0.5 <= x_coord < 1.5:
        #     return 0.5 * relu(lin + 1.5)**2 - 3 * relu(lin + 1.5) + 4.5
        # else:
        #     return 0

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
                 scale_tensor=[],
                 pos_encode=False,
                 sidelength=512,
                 fn_samples=None,
                 use_nyquist=True):
        super().__init__()

        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = Bsplines_form

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
                        trainable=True))

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

import torch
import torch.nn.functional as F
from torch import nn


class Bsplines_knots(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 is_first=False,
                 omega0=-0.2,
                 sigma0=0.5,
                 trainable=True):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        # self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def N_torch(self, i, k, x, t):
        """
        Recursive function (De Boor Algorithm) to calculate the value of the B-spline basis function using PyTorch.
        
        Parameters:
            i (int): the index of the control point.
            k (int): the degree of the B-spline.
            x (torch.Tensor): the points at which to evaluate the basis function.
            t (torch.Tensor): the knot vector.

        Returns:
            torch.Tensor: the values of the B-spline basis function at x.
        """
        if k == 0:
            return ((t[i] <= x) & (x < t[i + 1])).float()
        else:
            # Avoid division by zero
            denom1 = t[i + k] - t[i]
            denom2 = t[i + k + 1] - t[i + 1]

            # Calculate the first term in the recursion
            c1 = torch.zeros_like(x)
            valid1 = denom1 != 0
            c1[valid1] = (x[valid1] - t[i]) / denom1 * self.N_torch(i, k - 1, x, t)

            # Calculate the second term in the recursion
            c2 = torch.zeros_like(x)
            valid2 = denom2 != 0
            c2[valid2] = (t[i + k + 1] - x[valid2]) / denom2 * self.N_torch(
                i + 1, k - 1, x, t)

            return c1 + c2 

    def forward(self, input):
        lin = self.linear(input)
        knot_vec = torch.tensor([-1.5, -1.5, -1.5, -0.5, 0.5, 1.5, 1.5, 1.5], dtype=torch.float32)
        return self.N_torch(2, 2, lin, knot_vec)


class INR(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 outermost_linear=True,
                 first_omega_0=-0.2,
                 hidden_omega_0=-0.2,
                 scale=0.5,
                 scale_tensor=[],
                 pos_encode=False,
                 sidelength=512,
                 fn_samples=None,
                 use_nyquist=True):
        super().__init__()

        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = Bsplines_knots

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





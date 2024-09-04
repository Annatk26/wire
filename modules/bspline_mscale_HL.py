import torch
from torch import nn
import numpy as np
from modules.utils import build_montage

class Bsplines_form(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega0=-0.2,
        sigma0=6.0,
        init_weights=False,
        trainable=False,
    ):
        super().__init__()

        self.scale_0 = sigma0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init_weights:
            self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # self.linear.weight.uniform_(-20000 / self.in_features,
                #                              20000 / self.in_features)
                (self.linear.weight.normal_(mean=0.0, std=2 / (self.in_features)),)
            # else:
            #     self.linear.weight.normal_(mean=0.0, std=2/self.in_features)*np.sqrt(2/self.in_features)

    def quadratic_relu(self, x):
        return torch.nn.ReLU()(x) ** 2

    def forward(self, input):
        lin = self.linear(input)
        lin = lin / self.scale_0
        return (
            0.5 * self.quadratic_relu(lin + 1.5)
            - 1.5 * self.quadratic_relu(lin + 0.5)
            + 1.5 * self.quadratic_relu(lin - 0.5)
            - 0.5 * self.quadratic_relu(lin - 1.5)
        )


class Scaled_Bsplines_form(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega0=-0.2,
        sigma0=[],
        init_weights=False,
        trainable=False,
    ):
        super().__init__()
        self.scale_0 = sigma0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features

        self.scale_0 = nn.Parameter(self.scale_0.clone().detach(), trainable)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def quadratic_relu(self, x):
        return torch.nn.ReLU()(x) ** 2

    def quadratic_bspline(self, x, scale):
        x = x / scale
        return (
            0.5 * self.quadratic_relu(x + 1.5)
            - 1.5 * self.quadratic_relu(x + 0.5)
            + 1.5 * self.quadratic_relu(x - 0.5)
            - 0.5 * self.quadratic_relu(x - 1.5)
        )

    def forward(self, input):
        lin = self.linear(input)
        # Slice the output into parts
        split_size = (lin.size(2) - 256) // (len(self.scale_0) - 1)
        split_tensor_1 = lin[..., :256].clone().detach()
        split_tensor_2 = [
            lin[..., 256 + i * split_size : 256 + (i + 1) * split_size].clone().detach()
            for i in range(len(self.scale_0) - 1)
        ]
        # Apply different activations
        out_tensor = []
        out_tensor.append(self.quadratic_bspline(split_tensor_1, self.scale_0[0]))
        for i in range(len(self.scale_0) - 1):
            out_tensor.append(
                self.quadratic_bspline(split_tensor_2[i], self.scale_0[i + 1])
            )
        output = torch.cat(out_tensor, dim=2)
        return output


class INR(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        scaled_hidden_features,
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
        use_nyquist=True,
    ):
        super().__init__()

        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.net = []

        self.nonlin = Bsplines_form
        self.nonlin_first = Scaled_Bsplines_form
        self.scale_tensor: torch.Tensor = scale_tensor
        self.in_features = in_features
        self.complex = False
        self.pos_encode = False

        # hidden_layers = hidden_layers - 1
        self.net.append(
            self.nonlin_first(
                in_features,
                scaled_hidden_features,
                omega0=first_omega_0,
                sigma0=scale_tensor,
                is_first=True,
                trainable=False,
            )
        )

        self.net.append(
            self.nonlin(
                scaled_hidden_features,
                hidden_features,
                omega0=hidden_omega_0,
                sigma0=scale,
            )
        )

        for i in range(hidden_layers - 1):
            self.net.append(
                self.nonlin(
                    hidden_features,
                    hidden_features,
                    omega0=hidden_omega_0,
                    sigma0=scale,
                )
            )

        if outermost_linear:
            if self.complex:
                dtype = torch.cfloat
            else:
                dtype = torch.float

            final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)
            self.net.append(final_linear)
        else:
            self.net.append(
                self.nonlin(
                    hidden_features, out_features, omega0=hidden_omega_0, sigma0=scale
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords: torch.Tensor):
        output = self.net(coords)
        return output
    
    def forward_with_activations(self, coords, H, W, nfilters_vis='all'):
        atom_montages = []
    
        for idx in range(len(self.net)-1):
            layer_output = self.net[idx](coords)
            layer_images = layer_output.reshape(1, H, W, -1)[0]
            
            if nfilters_vis != 'all':
                layer_images = layer_images[..., 120:120+nfilters_vis]
                

            atoms = layer_images.detach().cpu().numpy().real
                
            atoms_min = atoms.min(0, keepdims=True).min(1, keepdims=True)
            atoms_max = atoms.max(0, keepdims=True).max(1, keepdims=True)
            
            signs = (abs(atoms_min) > abs(atoms_max))
            atoms = (1 - 2*signs)*atoms
            
            # Arrange them by variance
            atoms_std = atoms.std((0,1))
            std_indices = np.argsort(atoms_std)
            
            atoms = atoms[..., std_indices]
            
            atoms_min = atoms.min(0, keepdims=True).min(1, keepdims=True)
            atoms_max = atoms.max(0, keepdims=True).max(1, keepdims=True)
            
            atoms = (atoms - atoms_min)/np.maximum(1e-14, atoms_max - atoms_min)
            
            atoms[:, [0, -1], :] = 1
            atoms[[0, -1], :, :] = 1
            
            atoms_montage = build_montage(np.transpose(atoms, [2, 0, 1]))
            
            atom_montages.append(atoms_montage)
            coords = layer_output
            
        return atom_montages


import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class Bsplines_form(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            is_first=False,
            omega0=-0.2,  # a = 1.0
            sigma0=6.0,  # k = 10.0
            # scale_tensor=[],
            init_weights=False,
            trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        # self.scale_tensor = scale_tensor
        self.is_first = is_first

        self.in_features = in_features
        self.out_features = out_features
        # self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # self.linear.weight.uniform_(-20000 / self.in_features, 
                #                              20000 / self.in_features)
                self.linear.weight.normal_(mean=0.0, std=2/(self.in_features)), 
            # else:
            #     self.linear.weight.normal_(mean=0.0, std=2/self.in_features)*np.sqrt(2/self.in_features)

    def quadratic_relu(self, x):
        return torch.nn.ReLU()(x)**2

    def forward(self, input):
        lin = self.linear(self.scale_0*input)
        return (
            0.5 * self.quadratic_relu(lin + 1.5)
            - 1.5 * self.quadratic_relu(lin + 0.5)
            + 1.5 * self.quadratic_relu(lin - 0.5)
            - 0.5 * self.quadratic_relu(lin - 1.5)
        )


class INR(nn.Module):

    def __init__(self,
                 in_features,
                 scaled_hidden_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 outermost_linear=True,
                 first_omega_0=-0.2,
                 hidden_omega_0=-0.2,
                 scale=15.0,
                 scale_tensor=[],
                 pos_encode=False,
                 multiscale=True,
                 sidelength=512,
                 fn_samples=None,
                 use_nyquist=True):
        
        super().__init__()

        self.nonlin = Bsplines_form
        self.complex = False
        self.pos_encode = False
        self.scale_tensor = scale_tensor
        self.scale_weights = nn.Parameter(torch.ones((len(scale_tensor), )))
        # self.scale_tensor = scale_tensor

        self.net = []
        self.net.append(
                self.nonlin(in_features,
                            hidden_features,
                            omega0=first_omega_0,
                            sigma0=scale))
    
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
        all_outputs = []
        for i, scale in enumerate(self.scale_tensor):
            outputs = self.net(coords)
            all_outputs.append(outputs)
            # print(self.net(scaled_coords).shape)
        outputs = torch.stack(all_outputs, dim=0)
        return torch.sum(self.scale_weights.softmax(dim=0).view(-1,1,1,1) * outputs, dim=0)
       






# class Bsplines_form(nn.Module):

#     def __init__(
#             self,
#             in_features,
#             out_features,
#             bias=True,
#             is_first=False,
#             omega0=-0.2,  # a = 1.0
#             sigma0=6.0,  # k = 10.0
#             # scale_tensor=[],
#             init_weights=False,
#             trainable=False):
#         super().__init__()
#         self.omega_0 = omega0
#         self.scale_0 = sigma0
#         # self.scale_tensor = scale_tensor
#         self.is_first = is_first

#         self.in_features = in_features
#         self.out_features = out_features
#         # self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
#         self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

#         self.linear = nn.Linear(in_features, out_features, bias=bias)
#         if init_weights:
#             self.init_weights()

#     def init_weights(self):
#         with torch.no_grad():
#             if self.is_first:
#                 # self.linear.weight.uniform_(-20000 / self.in_features, 
#                 #                              20000 / self.in_features)
#                 self.linear.weight.normal_(mean=0.0, std=2/(self.in_features)), 
#             # else:
#             #     self.linear.weight.normal_(mean=0.0, std=2/self.in_features)*np.sqrt(2/self.in_features)

#     def quadratic_relu(self, x):
#         return torch.nn.ReLU()(x)**2

#     def forward(self, input):
#         lin = self.linear(input)
#         return (
#             0.5 * self.quadratic_relu(lin + 1.5)
#             - 1.5 * self.quadratic_relu(lin + 0.5)
#             + 1.5 * self.quadratic_relu(lin - 0.5)
#             - 0.5 * self.quadratic_relu(lin - 1.5)
#         )
    
# class Bspline_Subnet(nn.Module):

#     def __init__(
#             self,
#             in_features,
#             hidden_features,
#             hidden_layers,
#             out_features,
#             outermost_linear=True,
#             first_omega_0=-0.2,
#             hidden_omega_0=-0.2,
#             scale=15.0,
#             scale_tensor=[],
#             pos_encode=False,
#             sidelength=512,
#             fn_samples=None,
#             use_nyquist=True):
#         super().__init__()
#         self.net = []
#         self.complex = False
#         self.pos_encode = False
#         self.scale_tensor = scale_tensor

#         self.nonlin = Bsplines_form
#         self.scale_weights = nn.Parameter(torch.ones((len(scale_tensor), )))

#         self.net.append(
#                 self.nonlin(in_features,
#                             hidden_features,
#                             omega0=first_omega_0,
#                             sigma0=scale))
        
#         for i in range(hidden_layers):
#             self.net.append(
#                 self.nonlin(hidden_features,
#                             hidden_features,
#                             omega0=hidden_omega_0,
#                             sigma0=scale))

#         if outermost_linear:
#             if self.complex:
#                 dtype = torch.cfloat
#             else:
#                 dtype = torch.float

#             final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)
#             self.net.append(final_linear)
#         else:
#             self.net.append(
#                 self.nonlin(hidden_features,
#                             out_features,
#                             omega0=hidden_omega_0,
#                             sigma0=scale))

#         self.net = nn.Sequential(*self.net)

#     def forward(self, coords):
#         all_outputs = []
#         for i, scale in enumerate(self.scale_tensor):
#             scaled_coords = coords * scale
#             outputs = self.net(scaled_coords)
#             all_outputs.append(outputs)
#             # print(self.net(scaled_coords).shape)
#         outputs = torch.stack(all_outputs, dim=0)
#         return torch.sum(self.scale_weights.softmax(dim=0).view(-1,1,1,1) * outputs, dim=0)
#         # output = torch.sum(torch.stack([self.net(scale*coords) for scale in self.scale_tensor], dim=0)) 
#         # print(output.shape)
#         # # output = self.net(coords)
#         # return output


# class INR(nn.Module):

#     def __init__(self,
#                  in_features,
#                  scaled_hidden_features,
#                  hidden_features,
#                  hidden_layers,
#                  out_features,
#                  outermost_linear=True,
#                  first_omega_0=-0.2,
#                  hidden_omega_0=-0.2,
#                  scale=15.0,
#                  scale_tensor=[],
#                  pos_encode=False,
#                  multiscale=True,
#                  sidelength=512,
#                  fn_samples=None,
#                  use_nyquist=True):
#         super().__init__()

#         self.network = []
#         self.subnet = Bspline_Subnet
#         self.nonlin = Bsplines_form
        
#         self.network.append(self.subnet(in_features,
#                             hidden_features,
#                             hidden_layers,
#                             out_features,
#                             outermost_linear,
#                             first_omega_0,
#                             hidden_omega_0,
#                             scale,
#                             scale_tensor,
#                             pos_encode,
#                             multiscale,
#                             sidelength,
#                             fn_samples,
#                             use_nyquist))
#         # self.network.append(self.nonlin(len(scale_tensor)*out_features, out_features))
#         self.network = nn.Sequential(*self.network)


#     def forward(self, coords):
#         output = self.network(coords)
#         return output

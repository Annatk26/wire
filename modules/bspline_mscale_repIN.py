import torch
from torch import nn

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
            trainable=False):
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
                self.linear.weight.normal_(mean=0.0, std=2/(self.in_features)), 
            # else:
            #     self.linear.weight.normal_(mean=0.0, std=2/self.in_features)*np.sqrt(2/self.in_features)

    def quadratic_relu(self, x):
        return torch.nn.ReLU()(x)**2

    def forward(self, input):
        lin = self.linear(input)
        lin = self.scale_0 * lin
        return (
            0.5 * self.quadratic_relu(lin + 1.5)
            - 1.5 * self.quadratic_relu(lin + 0.5)
            + 1.5 * self.quadratic_relu(lin - 0.5)
            - 0.5 * self.quadratic_relu(lin - 1.5)
        )

class INR(nn.Module):

    def __init__(self,
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
                 use_nyquist=True):
        super().__init__()

        self.net = []

        self.nonlin = Bsplines_form
        self.scale_tensor = scale_tensor
        self.in_features = in_features

        self.complex = False
        self.pos_encode = False

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
        scales = self.scale_tensor.view(1, -1, 1, 1)
        expanded_coords = coords.unsqueeze(1)
        scaled_coords = scales * expanded_coords
        repeat_factor = int(self.in_features/(2*scales.size(1)))
        repeated_coords = scaled_coords.repeat(1, 1, 1, repeat_factor)
        # result = repeated_coords.permute(1, 0, 2, 3).reshape(coords.size(0), -1, 2*len(self.scale_tensor)*repeat_factor)
        result = repeated_coords.permute(1, 0, 2, 3).reshape(coords.size(0), -1, 2*scales.size(1)*repeat_factor)
        # all_rows = torch.stack([result[:, :, i, :].squeeze(0).flatten() for i in range(result.shape[2])]).unsqueeze(0)
        output = self.net(result)
        return output

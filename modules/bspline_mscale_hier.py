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
        trainable=False
    ):
        super().__init__()

        self.scale_0 = sigma0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1, device='cuda'), trainable)
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
        multiscale=True,
        sidelength=512,
        fn_samples=None,
        use_nyquist=True,
    ):
        super().__init__()

        self.stages = nn.ModuleList()
        self.linears = []
        self.nonlin = Bsplines_form
        self.scale_tensor: torch.Tensor = scale_tensor
        self.complex = False
        self.pos_encode = False
        self.num_stages = len(scale_tensor)

        for stage in range(self.num_stages):
            layers = []
            layers.append(
                self.nonlin(
                    in_features, 
                    hidden_features, 
                    omega0=first_omega_0,
                    sigma0=scale_tensor[stage],
                )
            )

            layers.append(
                self.nonlin(
                    hidden_features * 2 if stage != 0 else hidden_features, 
                    hidden_features, 
                    omega0=hidden_omega_0,
                    sigma0=scale_tensor[stage],
                )
            )
                
            for _ in range(hidden_layers-1):
                layers.append(
                    self.nonlin(
                        hidden_features, 
                        hidden_features, 
                        omega0=hidden_omega_0,
                        sigma0=scale_tensor[stage],
                    )
                )
            
            self.stages.append(nn.Sequential(*layers))
            self.linears.append(nn.Linear(hidden_features, out_features))

    def forward(self, coords):
        x = coords
        outputs = []
        for stage in range(self.num_stages):
            if stage == 0:
                for i in range(len(self.stages[stage])):
                    x = self.stages[stage][i](x)
            else:
                x_in = self.stages[stage][0](coords)
                x_HL = self.stages[stage][1](torch.cat((x_in, x), dim=2))
                x = self.stages[stage][2](x_HL)
            self.linears[stage] = self.linears[stage].to(x.device)
            outputs.append(self.linears[stage](x))
        return torch.stack(outputs, dim=0).sum(dim=0)

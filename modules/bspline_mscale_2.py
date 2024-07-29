import torch
from torch import nn
class Bsplines_form(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            is_first=False,
            omega0=-0.2,  # a = 1.0
            # sigma0=6.0,  # k = 10.0
            # scale_tensor=[],
            init_weights=False,
            trainable=False):
        super().__init__()
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features

        # self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)
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

    def forward(self, input, scale):
        lin = self.linear(input)
        lin = lin / scale
        return (
            0.5 * self.quadratic_relu(lin + 1.5)
            - 1.5 * self.quadratic_relu(lin + 0.5)
            + 1.5 * self.quadratic_relu(lin - 0.5)
            - 0.5 * self.quadratic_relu(lin - 1.5)
        )

class AdaptiveScaleCombiner(nn.Module):
    def __init__(self, num_scales, out_features, image_size, type):
        super().__init__()
        self.num_scales = num_scales
        self.out_features = out_features
        self.image_size = image_size
        self.type = type

        # Adaptive scale weighting
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        # Frequency-based combination
        self.freq_mlp = nn.Sequential(
                    nn.Linear(num_scales * out_features, 128),
                    nn.ReLU(),
                    nn.Linear(128, out_features)
                    )   
        # Attention mechanism
        # self.attention_dim = 6  # Choose a dimension that's divisible by common head numbers (1, 2, 4, 8)
        # self.attention_proj_in = nn.Linear(out_features, self.attention_dim)
        # self.attention = nn.MultiheadAttention(embed_dim=self.attention_dim, num_heads=2)
        # self.attention_proj_out = nn.Linear(self.attention_dim, out_features)
        
        # Final refinement
        self.refine = nn.Sequential(
            nn.Linear(out_features, 128),
            nn.ReLU(),
            nn.Linear(128, out_features)
        )

    def forward(self, outputs, type):
        # 1. Adaptive Scale Weighting
        if type == 'scale_weights':
            output = [w * out for w, out in zip(self.scale_weights, outputs)]
            final_output = torch.stack(output).sum(dim=0)
        # 2. Frequency-based Combination
        elif type == 'freq_combine':
            concat_outputs = torch.cat(outputs, dim=-1)
            final_output = self.freq_mlp(concat_outputs)
        elif type == 'both':
            weighted_outputs = [w * out for w, out in zip(self.scale_weights, outputs)]
            concat_outputs = torch.cat(weighted_outputs, dim=-1)
            freq_combined = self.freq_mlp(concat_outputs)
            final_output = self.refine(freq_combined)    
        return final_output
    
class INR(nn.Module):
    
    # def combine_scales(self, outputs):
    #     return torch.stack(outputs).sum(dim=0)

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
                 multiscale=True,
                 sidelength=512,
                 fn_samples=None,
                 use_nyquist=True):
        
        super().__init__()

        self.nonlin = Bsplines_form
        self.complex = False
        self.pos_encode = False
        self.scale0 = scale
        self.scale_tensor = scale_tensor
        self.outermost_linear = outermost_linear
        self.combine_scales = AdaptiveScaleCombiner(len(scale_tensor), out_features, sidelength, 'both')

        self.net = []
        self.net.append(
                self.nonlin(in_features,
                            hidden_features,
                            omega0=first_omega_0,
                            # sigma0=scale
                            ))
    
        for i in range(int(hidden_layers)):
            self.net.append(
                self.nonlin(hidden_features,
                            hidden_features,
                            omega0=hidden_omega_0,
                            # sigma0=scale
                            ))
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
                            # sigma0=scale
                            ))
        
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        output = []
        for scale in self.scale_tensor:
            if self.outermost_linear:
                out = x
                for layer in self.net[:-1]:  # All layers except the last one
                    out = layer(out, scale)
                out = self.net[-1](out)  # Last layer (either linear or Bsplines_form)
                output.append(out)
            else:
                out = x
                for layer in self.net:
                    out = layer(out, scale)
                output.append(out)
        return self.combine_scales(output, 'freq_combine')
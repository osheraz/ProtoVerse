import torch
from torch import nn, Tensor
from hydra.utils import instantiate
from protoverse.agents.common.common import NormObsBase
from protoverse.utils import model_utils


def build_mlp(config, num_in: int, num_out: int):
    """
    Build a feed-forward MLP from a config that looks like:
      config.layers = [
        { units: 1024, activation: "tanh", use_layer_norm: false },
        { units: 512,  activation: "tanh", use_layer_norm: false },
      ]
    - LayerNorm is applied ONLY to the FIRST hidden layer if requested.
    - No activation on the final (output) layer.
    """
    indim = num_in
    layers = []
    for i, layer in enumerate(config.layers):
        layers.append(nn.Linear(indim, layer.units))
        if layer.use_layer_norm and i == 0:  # only first hidden layer
            layers.append(nn.LayerNorm(layer.units))
        layers.append(model_utils.get_activation_func(layer.activation))
        indim = layer.units

    layers.append(nn.Linear(indim, num_out))  # final linear, no activation
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """
    Simple MLP that can take either:
      - a raw tensor of shape [B, num_in], or
      - a dict and read input_dict[config.obs_key].
    """

    def __init__(self, config, num_in: int, num_out: int):
        super().__init__()
        self.config = config
        self.mlp = build_mlp(self.config, num_in, num_out)

    def forward(self, input_dict, *args, **kwargs):
        if isinstance(input_dict, torch.Tensor):
            return self.mlp(input_dict)  # already a tensor
        return self.mlp(input_dict[self.config.obs_key])  # dict -> pick key


class MLP_WithNorm(NormObsBase):
    """
    MLP that first normalizes observations using NormObsBase:
      - super().forward(tensor) returns normalized obs (and handles stateful stats).
      - If return_norm_obs=True, also returns the normalized tensor for logging.
    Expects input_dict[self.config.obs_key] to be a tensor [B, num_in].
    """

    def __init__(self, config, num_in: int, num_out: int):
        super().__init__(config, num_in, num_out)
        self.mlp = build_mlp(self.config, num_in, num_out)

    def forward(self, input_dict, return_norm_obs=False):
        obs = super().forward(
            input_dict[self.config.obs_key]
        )  # normalize via NormObsBase
        outs: Tensor = self.mlp(obs)

        if return_norm_obs:
            return {"outs": outs, f"norm_{self.config.obs_key}": obs}
        else:
            return outs


class MultiHeadedMLP(nn.Module):
    """
    Multi-input MLP:
      - Instantiates a set of "input_models" from Hydra config (e.g., Flatten, MLP_WithNorm, etc.).
      - Concatenates their outputs along the last dim.
      - Feeds the concatenated features to a trunk MLP.

    Config expectations:
      config.input_models: dict[str -> hydra-target with .num_out attribute]
      config.trunk: hydra-target (e.g., MLP) that accepts num_in=<sum of num_out over heads>

    Forward:
      - If return_norm_obs=True and a submodel supports it, returns both "outs" and per-head normalized tensors.
    """

    def __init__(self, config, num_out: int):
        super().__init__()
        self.config = config
        self.num_out = num_out

        # Instantiate each head; accumulate total feature size.
        input_models = {}
        self.feature_size = 0
        for key, input_cfg in self.config.input_models.items():
            model = instantiate(input_cfg)  # must expose model.num_out
            input_models[key] = model
            self.feature_size += model.num_out
        self.input_models = nn.ModuleDict(input_models)

        # Trunk receives concatenated features; num_in is determined here.
        self.trunk: MLP = instantiate(self.config.trunk, num_in=self.feature_size)

    def forward(self, input_dict, return_norm_obs=False):
        if return_norm_obs:
            norm_obs = {}  # collect normalized inputs per head
        outs = []

        # Run each head; support both simple heads (tensor out) and heads that
        # return dicts when return_norm_obs=True (e.g., MLP_WithNorm).
        for key, model in self.input_models.items():
            out = model(input_dict, return_norm_obs=return_norm_obs)

            if return_norm_obs:
                # Expect a dict: {"outs": <tensor>, f"norm_{obs_key}": <tensor>}
                out, norm_obs[f"norm_{model.config.obs_key}"] = (
                    out["outs"],
                    out[f"norm_{model.config.obs_key}"],
                )
            outs.append(out)  # each is [B, model.num_out]

        outs = torch.cat(outs, dim=-1)  # -> [B, sum(num_out_i)]
        outs: Tensor = self.trunk(outs)  # final features -> output

        if return_norm_obs:
            ret_dict = {**{"outs": outs}, **norm_obs}
            return ret_dict
        else:
            return outs

import math

import torch


def _pre_hook(state_dict, prefix, local_metadata, strict,
              missing_keys, unexpected_keys, error_msgs):
    """Perform pre-hook in load_state_dict for backward compatibility.

    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.

    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class AbsolutePositionalEncoding(torch.nn.Module):
    """Positional encoding.
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    """

    def __init__(self, d_model):
        """Construct an PositionalEncoding object."""
        super(AbsolutePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self._register_load_state_dict_pre_hook(_pre_hook)

    def forward(self, olen):
        pe = torch.zeros(olen, self.d_model)
        position = torch.arange(0, olen, dtype=torch.float32).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) *
        #                      -(math.log(10000.0) / self.d_model))
        div_term = 1 / (10000.0 ** (torch.arange(0, self.d_model, 2, dtype=torch.float32) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

class RelativePositionalEncoding(torch.nn.Module):
    """Positional encoding.
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    """

    def __init__(self, d_model):
        """Construct an PositionalEncoding object."""
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self._register_load_state_dict_pre_hook(_pre_hook)

    def forward(self, olen, ilen):
        pe = torch.zeros(olen, self.d_model)
        position = torch.arange(0, olen, dtype=torch.float32).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) *
        #                      -(math.log(10000.0) / len))
        div_term = 1 / (10000.0 ** (torch.arange(0, self.d_model, 2, dtype=torch.float32) / ilen))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe


class TimeVaryingMetaTemplate(torch.nn.Module):
    def __init__(self, d_model):
        """Construct an PositionalEncoding object."""
        super(TimeVaryingMetaTemplate, self).__init__()

        self.ape = AbsolutePositionalEncoding(d_model)
        self.rpe = RelativePositionalEncoding(d_model)

    def forward(self, olen, ilen):
        ape = self.ape(olen)
        rpe = self.rpe(olen, ilen)
        TVMT = torch.cat((ape, rpe), 1)
        return TVMT.transpose(0, 1)
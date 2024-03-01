import torch, numpy as np
from torch.nn import *
from config import config

class Net(Module):
    def __init__(self):
        super().__init__()

        self.device = torch.device(config.device)
        self.lr = config.opt_lr
        self.alpha_h = config.alpha_h

    def save(self, file='model.pt'):
        torch.save(self.state_dict(), file)

    def load(self, file='model.pt'):
        self.load_state_dict(torch.load(file, map_location=self.device))

    def copy_weights(self, other, rho):
        params_other = list(other.parameters())
        params_self  = list(self.parameters())

        for i in range( len(params_other) ):
            val_self  = params_self[i].data
            val_other = params_other[i].data
            val_new   = rho * val_other + (1-rho) * val_self

            params_self[i].data.copy_(val_new)

    def set_lr(self, lr):
        self.lr = lr

        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def set_alpha_h(self, alpha_h):
        self.alpha_h = alpha_h

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())

    def reset_state(self, batch_mask=None):
        pass

    def clone_state(self, other):
        pass

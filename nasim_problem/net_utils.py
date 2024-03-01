from numba import jit
import torch, numpy as np
import torch_geometric

def segmented_sample(probs, splits):
    probs_split = torch.split(probs, splits)
    samples = [torch.multinomial(x, 1) for x in probs_split]
    
    return torch.cat(samples)

def flatten(t):
    return [item for sublist in t for item in sublist]

# @jit(nopython=True)
# @torch.jit.script
def compute_v_target(r, v_, d, gamma, ppo_t, batch, clip_to_zero):
    d = torch.tensor(d, dtype=torch.float32)
    r = torch.tensor(r, dtype=torch.float32)

    v_target = torch.zeros(ppo_t, batch)

    if clip_to_zero:
        for t_reversed in range(int(ppo_t)):
            t = ppo_t - 1 - t_reversed

            if t == ppo_t - 1:
                v_target[t] = r[t] + (1 - d[t]) * gamma * torch.clamp(v_, 0., None)

            else:
                v_target[t] = r[t] + (1 - d[t]) * gamma * torch.clamp(v_target[t+1], 0., None)

    else:
        for t_reversed in range(int(ppo_t)):
            t = ppo_t - 1 - t_reversed

            if t == ppo_t - 1:
                v_target[t] = r[t] + (1 - d[t]) * gamma * v_

            else:
                v_target[t] = r[t] + (1 - d[t]) * gamma * v_target[t+1]

    return v_target

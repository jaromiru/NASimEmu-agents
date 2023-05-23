from nasimemu.nasim.envs.host_vector import HostVector

import torch, numpy as np
import torch_geometric

from numba import jit

from torch.nn import *
from torch_geometric.data import Data, Batch

from rl import a2c, ppo
from config import config
from net import Net

from graph_nns import *

import wandb

def segmented_sample(probs, splits):
    probs_split = torch.split(probs, splits)
    samples = [torch.multinomial(x, 1) for x in probs_split]
    
    return torch.cat(samples)

def flatten(t):
    return [item for sublist in t for item in sublist]

@jit(nopython=True)
def compute_v_target(r, v_, d, gamma, ppo_t, batch):
    v_target = np.zeros( (ppo_t, batch) )

    for t_reversed in range(ppo_t):
        t = ppo_t - 1 - t_reversed

        if t == ppo_t - 1:
            v_target[t] = r[t] + (1 - d[t]) * gamma * v_

        else:
            v_target[t] = r[t] + (1 - d[t]) * gamma * v_target[t+1]

    return v_target

@jit(nopython=True)
def compute_q_target(r, q_, d, gamma, ppo_t, batch):
    q_target = np.zeros( (ppo_t, batch) )

    for t_reversed in range(ppo_t):
        t = ppo_t - 1 - t_reversed
        
        if t == ppo_t - 1:
            q_target[t] = r[t] + (1 - d[t]) * gamma * np.clip(q_, 0., None) # max of q_ and 0.

        else:
            q_target[t] = r[t] + (1 - d[t]) * gamma * np.clip(q_target[t+1], 0., None)

    return q_target

class NASimNetMLP(Net):

    def __init__(self):
        super().__init__()

        self.MAX_ROWS = 30

        self.mlp = Sequential( Linear(self.MAX_ROWS * (config.node_dim - 1), config.emb_dim), LeakyReLU())#,
                               #Linear(config.emb_dim, config.emb_dim), LeakyReLU())

        self.action_select = Linear(config.emb_dim, self.MAX_ROWS * config.action_dim)  

        self.value_function = Linear(config.emb_dim, 1) 
        self.q_opt_function = Linear(config.emb_dim, 1) # not used currently
        self.q_opt_function.requires_grad_(False) # disable learning

        self.opt = torch.optim.AdamW(self.parameters(), lr=config.opt_lr, weight_decay=config.opt_l2)
        # self.opt = torch.optim.RAdam(self.parameters(), lr=config.opt_lr, weight_decay=config.opt_l2)
        self.to(self.device)

        self.force_continue = False

    def prepare_batch(self,s_batch):
        batch = []
        node_index = []
        action_mask = []

        for s in s_batch:
            s = s[:-1] # remove result

            am = np.zeros(config.action_dim * self.MAX_ROWS) # create action mask
            am[len(s) * config.action_dim:] = 1
            action_mask.append(am)

            s = np.pad(s, ( (0, self.MAX_ROWS - len(s)), (0, 0) )) # pad with zeros

            node_index.append([HostVector(x).address for x in s]) # create node_index

            s = s.flatten() # flat for MLP

            batch.append(s)

        node_index = np.array(node_index)
        action_mask = np.array(action_mask)
        action_mask = torch.tensor(action_mask, dtype=torch.bool, device=self.device)

        batch = np.array(batch)
        batch = torch.tensor(batch, dtype=torch.float32, device=config.device)

        return batch, node_index, action_mask

    def forward(self, s_batch, only_v=False, complete=False, force_action=None):
        batch, node_index, action_mask = self.prepare_batch(s_batch)

        # process state
        x = self.mlp(batch)

        # decode value
        value = self.value_function(x)
        q_val = self.q_opt_function(x)

        if only_v:
            return value, q_val

        # select an action & node
        def sample_action(x):
            out_action = self.action_select(x)
            out_action[action_mask] = -np.inf # apply action mask - disable actions for non-existent nodes
            action_softmax = torch.distributions.Categorical( torch.softmax(out_action, dim=1) )

            if force_action is not None:
                action_selected = force_action
            else:
                action_selected = action_softmax.sample()

            a_prob = action_softmax.probs.gather(1, action_selected.view(-1, 1))

            return a_prob, action_selected

        a_prob, action_selected = sample_action(x)
        
        a_id = (action_selected % config.action_dim).clone().cpu().numpy()
        n_id = (action_selected // config.action_dim).clone().cpu().numpy()
        targets = node_index[np.arange(len(node_index)), n_id].reshape(-1, 2) 

        # total probability of the action is the product of probabilites of selecting a node and a particular action on this node
        tot_prob = a_prob
        env_actions = action_selected.clone().cpu().numpy()

        actions = list(zip(targets, a_id))
        raw_actions = (action_selected.detach())

        return actions, value, q_val, tot_prob, raw_actions

    def update(self, trace, target_net=None):
        sx, a, a_q, r, sx_, d = zip(*trace)

        s = np.empty((config.ppo_t, config.batch), dtype=np.object)
        s[:,:] = sx

        s_ = np.empty((config.ppo_t, config.batch), dtype=np.object)
        s_[:,:] = sx_

        r = np.vstack(r)
        d = np.vstack(d)

        a = torch.cat(a)

        if target_net is None:
            target_net = self

        # be carefull here: d_true will not work with PPO (as currently implemented)
        # the true next state is given only here as s_, but inside the sequence itself, we use 
        # plain next state - this is OK if all d_true == d, but wrong otherwise

        v_, q_ = target_net(s_[-1], only_v=True)
        v_ = v_.detach().cpu().numpy().flatten()    # TODO: this could possibly be done without leaving GPU
        q_ = q_.detach().cpu().numpy().flatten()

        v_target = compute_v_target(r, v_, d, config.gamma, config.ppo_t, config.batch)
        q_target = compute_q_target(r, q_, d, config.gamma, config.ppo_t, config.batch)

        # print(d)
        # print(r)
        # print("q_", q_)
        # print("q_target", q_target)
        # exit()

        s = np.concatenate(s)
        v_target = torch.tensor(np.concatenate(v_target), dtype=torch.float32, device=self.device)
        q_target = torch.tensor(np.concatenate(q_target), dtype=torch.float32, device=self.device)
        a_q = torch.tensor(np.concatenate(a_q), dtype=torch.bool, device=self.device)

        # print(f"\nv={v_.mean():.2f}, v_t={v_target.mean():.2f} / q={q_.mean():.2f}, q_t={q_target.mean():.2f}")
        wandb.log(dict(v=v_.mean(), v_t=v_target.mean(), q=q_.mean(), q_t=q_target.mean()), commit=False)

        # Todo: subnets should not be counted
        num_actions = torch.tensor([x[0].shape[0] * config.action_dim for x in s], dtype=torch.float32, device=self.device)
        return ppo(s, a, a_q, v_target, q_target, self, config.gamma, config.alpha_v, config.alpha_q, self.alpha_h, config.ppo_k, config.ppo_eps, config.v_range, num_actions )

    def _update(self, loss):
        self.opt.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), config.opt_max_norm) # clip the gradient norm
        self.opt.step()

        return norm        

    # the net will be forced not to issue TerminalAction
    def set_force_continue(self, force):
        self.force_continue = force

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
from .net_utils import *

import wandb

class NASimNetMLP_LSTM(Net):

    def __init__(self):
        super().__init__()

        self.MAX_ROWS = 50

        self.mlp = Sequential( Linear(self.MAX_ROWS * (config.node_dim - 1), config.emb_dim), LeakyReLU())

        self.gru = GRU(input_size=config.emb_dim, hidden_size=config.emb_dim)
        self.hidden = None

        self.action_select = Linear(config.emb_dim, self.MAX_ROWS * config.action_dim)  
        self.value_function = Linear(config.emb_dim, 1) 

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
        x, self.hidden = self.gru(x.unsqueeze(0), self.hidden)
        x = x.squeeze(0)

        # decode value
        value = self.value_function(x)

        if only_v:
            return value

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

        # return complete probs for debug
        # if complete:
        #     # node probs
        #     node_activation = self.node_select(x)
        #     subnet_nodes = batch.x[:, 0] == 1
        #     node_activation[subnet_nodes] = -np.inf # subnet nodes cannot be selected
        #     node_softmax = torch_geometric.utils.softmax(node_activation.flatten(), batch_ind)

        #     # action probs
        #     out_action = self.action_select(x)
        #     action_softmax = torch.softmax(out_action, dim=1)

        #     return node_softmax, action_softmax, value, q_val

        # TODO: mask actions...
        a_prob, action_selected = sample_action(x)
        
        a_id = (action_selected % config.action_dim).clone().cpu().numpy()
        n_id = (action_selected // config.action_dim).clone().cpu().numpy()
        targets = node_index[np.arange(len(node_index)), n_id].reshape(-1, 2) 

        # total probability of the action is the product of probabilites of selecting a node and a particular action on this node
        tot_prob = a_prob
        env_actions = action_selected.clone().cpu().numpy()

        if not self.force_continue:
            terminate = (value.detach() <= 0.).view(-1, 1)
            env_actions[terminate.flatten().cpu().numpy()] = -1
            tot_prob[terminate] = .5 # don't update probabilities when terminated (also, don't put 0. there, when it goes through torch.log, it throws nan errors)

        actions = list(zip(targets, a_id))
        raw_actions = (action_selected.detach())

        return actions, value, tot_prob, raw_actions

    # def update(self, trace, target_net=None):
    def update(self, trace, target_net=None, hidden_s0=None):
        sx, a, a_cnt, r, sx_, d = zip(*trace)

        s = np.empty((config.ppo_t, config.batch), dtype=object)
        s[:,:] = sx

        s_ = np.empty((config.ppo_t, config.batch), dtype=object)
        s_[:,:] = sx_

        r = np.vstack(r)
        d = np.vstack(d)

        # a = torch.cat(a)

        # if target_net is None:
        #     target_net = self

        # be carefull here: d_true will not work with PPO (as currently implemented)
        # the true next state is given only here as s_, but inside the sequence itself, we use 
        # plain next state - this is OK if all d_true == d, but wrong otherwise

        v_ = target_net(s_[-1], only_v=True)
        v_ = v_.detach().flatten()

        v_target = compute_v_target(r, v_, d, config.gamma, config.ppo_t, config.batch, config.use_a_t)

        # print(d)
        # print(r)
        # print("q_", q_)
        # print("q_target", q_target)
        # exit()

        # s = np.concatenate(s)
        v_target = v_target.flatten()
        a_cnt = torch.tensor(np.concatenate(a_cnt), dtype=torch.bool, device=self.device)

        # print(f"\nv={v_.mean():.2f}, v_t={v_target.mean():.2f} / q={q_.mean():.2f}, q_t={q_target.mean():.2f}")
        wandb.log(dict(v=v_.mean(), v_t=v_target.mean()), commit=False)

        # Todo: subnets should not be counted
        return ppo(s, a, a_cnt, d, v_target, self, config.gamma, config.alpha_v, self.alpha_h, config.ppo_k, config.ppo_eps, config.use_a_t, config.v_range, lstm=True, hidden_s0=hidden_s0)

    def _update(self, loss):
        self.opt.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), config.opt_max_norm) # clip the gradient norm
        self.opt.step()

        return norm        

    # the net will be forced not to issue TerminalAction
    def set_force_continue(self, force):
        self.force_continue = force


    def reset_state(self, batch_mask=None):
        if self.hidden is not None:
            if batch_mask is None:  # reset all
                self.hidden = None

            else:
                # reset_idx = np.where(batch_mask)
                zeros = torch.zeros_like(self.hidden, device=config.device)
                condition = torch.zeros_like(self.hidden, dtype=torch.bool, device=config.device) # TODO: not optimal...
                condition[0, batch_mask] = True
                self.hidden = torch.where(condition, zeros, self.hidden) # reset the marked places

    def clone_state(self, other):
        if other.hidden is None:
            self.hidden = None

        else:
            self.hidden = other.hidden.clone().detach()

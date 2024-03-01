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

class NASimNetGNN(Net):
    def __init__(self):
        super().__init__()

        self.embed_node = Sequential( Linear(config.node_dim + config.pos_enc_dim, config.emb_dim), LeakyReLU() )
        self.gnn = MultiMessagePassingWithGlobalNode(config.mp_iterations)

        # self.node_select = Sequential(Linear(config.emb_dim, config.emb_dim), LeakyReLU(), Linear(config.emb_dim, 5)) # node features -> node probability for all 5 actions
        # self.action_select = Sequential(Linear(config.emb_dim * 2, config.emb_dim), LeakyReLU(), Linear(config.emb_dim, 5))  # global features -> 5 actions
        # self.value_function = Sequential(Linear(config.emb_dim * 2, config.emb_dim), LeakyReLU(), Linear(config.emb_dim, 1)) # global features -> state value

        self.node_select = Linear(config.emb_dim, 1) # node features -> node probability
        self.action_select = Linear(config.emb_dim, config.action_dim)  # node features -> actions

        self.value_function = Linear(config.emb_dim, 1) # global features -> state value

        self.opt = torch.optim.AdamW(self.parameters(), lr=config.opt_lr, weight_decay=config.opt_l2)
        # self.opt = torch.optim.RAdam(self.parameters(), lr=config.opt_lr, weight_decay=config.opt_l2)
        self.to(self.device)

        self.force_continue = False

    @staticmethod
    def prepare_batch(s_batch):
        node_feats, edge_index, node_index, pos_index = zip(*s_batch)

        node_feats = [torch.tensor(x, dtype=torch.float32, device=config.device) for x in node_feats]
        # edge_attr  = [torch.tensor(x, dtype=torch.float32, device=config.device) for x in edge_attr]
        edge_index = [torch.tensor(x, dtype=torch.int64, device=config.device) for x in edge_index]

        # create batch
        data = [Data(x=node_feats[i], edge_index=edge_index[i]) for i in range( len(s_batch) )]
        data_lens = [x.num_nodes for x in data]
        batch = Batch.from_data_list(data)
        batch_ind = batch.batch.to(config.device) # graph indices in the batch

        #TODO node index requires concat
        node_index = np.concatenate(node_index)
        pos_index = torch.tensor(np.concatenate(pos_index)).to(config.device)

        return data, data_lens, batch, batch_ind, node_index, pos_index

    def forward(self, s_batch, only_v=False, complete=False, force_action=None):
        data, data_lens, batch, batch_ind, node_index, pos_index = self.prepare_batch(s_batch)
        x = batch.x

        # process state
        pos_enc = positional_encoding(pos_index, dim=config.pos_enc_dim)
        x = torch.cat([x, pos_enc], dim=1)    # add positional encoding

        x = self.embed_node(x)
        xg_init = None # TODO
        x, x_pooled = self.gnn(x, xg_init, batch.edge_attr, batch.edge_index, batch_ind, batch.num_graphs, data_lens)

        # decode value
        value = self.value_function(x_pooled)

        if only_v:
            return value

        def sample_action(x, n_index):
            out_action = self.action_select(x[n_index])

            action_softmax = torch.distributions.Categorical( torch.softmax(out_action, dim=1) )

            if force_action is not None:
                action_selected = force_action[1]
            else:
                action_selected = action_softmax.sample()

            a_prob = action_softmax.probs.gather(1, action_selected.view(-1, 1))

            return a_prob, action_selected

        def sample_node(x, batch):
            node_activation = self.node_select(x)

            subnet_nodes = batch.x[:, 0] == 1
            node_activation[subnet_nodes] = -np.inf # subnet nodes cannot be selected

            node_softmax = torch_geometric.utils.softmax(node_activation.flatten(), batch_ind)

            if force_action is not None:
                node_selected = force_action[0]
            else:
                node_selected = segmented_sample(node_softmax, data_lens)

            # get proper node probability indexes
            data_starts = np.concatenate( ([0], data_lens[:-1]) )
            data_starts = torch.tensor(data_starts, device=self.device, dtype=torch.int64)
            n_index = torch.cumsum(data_starts, 0) + node_selected
            n_prob = node_softmax[n_index].view(-1, 1)

            return n_prob, n_index, node_selected

        # return complete probs for debug
        if complete:
            # node probs
            node_activation = self.node_select(x)
            subnet_nodes = batch.x[:, 0] == 1
            node_activation[subnet_nodes] = -np.inf # subnet nodes cannot be selected
            node_softmax = torch_geometric.utils.softmax(node_activation.flatten(), batch_ind)

            # action probs
            out_action = self.action_select(x)
            action_softmax = torch.softmax(out_action, dim=1)

            return node_softmax, action_softmax, value, q_val

        # select an action & node
        n_prob, n_index, node_selected = sample_node(x, batch)
        a_prob, action_selected = sample_action(x, n_index)

        # total probability of the action is the product of probabilites of selecting a node and a particular action on this node
        tot_prob = a_prob * n_prob
        env_actions = action_selected.clone().cpu().numpy()

        if not self.force_continue:
            terminate = (value.detach() <= 0.).view(-1, 1)
            env_actions[terminate.flatten().cpu().numpy()] = -1
            tot_prob[terminate] = .5 # don't update probabilities when terminated (also, don't put 0. there, when it goes through torch.log, it throws nan errors)

        targets = node_index[n_index.cpu()].reshape(-1, 2)
        actions = list(zip(targets, env_actions))

        raw_actions = (node_selected.detach(), action_selected.detach())

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

        a0, a1 = zip(*a)
        a = ( torch.cat(a0), torch.cat(a1) )

        if target_net is None:
            target_net = self

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

        s = np.concatenate(s)
        v_target = v_target.flatten()
        a_cnt = torch.tensor(np.concatenate(a_cnt), dtype=torch.bool, device=self.device)

        # print(f"\nv={v_.mean():.2f}, v_t={v_target.mean():.2f} / q={q_.mean():.2f}, q_t={q_target.mean():.2f}")
        wandb.log(dict(v=v_.mean(), v_t=v_target.mean()), commit=False)

        return ppo(s, a, a_cnt, d, v_target, self, config.gamma, config.alpha_v, self.alpha_h, config.ppo_k, config.ppo_eps, config.use_a_t, config.v_range)

    def _update(self, loss):
        self.opt.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), config.opt_max_norm) # clip the gradient norm
        self.opt.step()

        return norm        

    # the net will be forced not to issue TerminalAction
    def set_force_continue(self, force):
        self.force_continue = force

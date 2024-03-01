import torch, torch_geometric, math
from torch.nn import *
from torch_geometric.nn import MessagePassing, GlobalAttention, TransformerConv, GATv2Conv, LayerNorm, global_mean_pool, global_max_pool
from torch.nn.functional import leaky_relu

from config import config

def positional_encoding(pos, dim):
    w = torch.exp(torch.arange(0, dim, 2, device=pos.device) * (-math.log(10000.0) / dim))
    pos_w = pos.unsqueeze(1) * w

    pe = torch.zeros(len(pos), dim, device=pos.device)
    pe[:, 0::2] = torch.sin(pos_w)
    pe[:, 1::2] = torch.cos(pos_w)

    return pe

# ----------------------------------------------------------------------------------------
class MultiMessagePassingWithAttention(Module):
    def __init__(self, steps):
        super().__init__()

        self.gnns = ModuleList( [GraphNet() for i in range(steps)] )
        self.att  = ModuleList( [GATv2Conv(config.emb_dim, config.emb_dim, add_self_loops=False, heads=3, concat=False) for i in range(steps - 1)] )
        self.pooling = GlobalPooling()

        self.steps = steps

    def forward(self, x, edge_attr, edge_index, batch_ind, num_graphs, data_lens):
        x_att = torch.zeros(len(x), config.emb_dim, device=config.device)  # this can encode context

        edge_complete = complete_graph(data_lens)

        for i in range(self.steps):
            x = self.gnns[i](x, edge_attr, edge_index, x_att, batch_ind)
            if i < self.steps - 1:
                x_att = leaky_relu( self.att[i](x, edge_complete) )

        x_global = self.pooling(x, batch_ind)
        return x, x_global

# ----------------------------------------------------------------------------------------
class MultiMessagePassingWithTransformer(Module):
    def __init__(self, steps):
        super().__init__()

        self.gnns = ModuleList( [GraphNet() for i in range(steps)] )
        self.att  = ModuleList( [TransformerConv(config.emb_dim, config.emb_dim, heads=3, concat=False) for i in range(steps - 1)] )
        self.pooling = GlobalPooling()

        self.steps = steps

    def forward(self, x, edge_attr, edge_index, batch_ind, num_graphs, data_lens):
        x_att = torch.zeros(len(x), config.emb_dim, device=config.device)  # this can encode context

        edge_complete = complete_graph(data_lens)

        for i in range(self.steps):
            x = self.gnns[i](x, edge_attr, edge_index, x_att, batch_ind)
            if i < self.steps - 1:
                x_att = leaky_relu( self.att[i](x, edge_complete) )

        x_global = self.pooling(x, batch_ind)
        return x, x_global

# ----------------------------------------------------------------------------------------
class MultiMessagePassingWithGlobalNode(Module):
    def __init__(self, steps):
        super().__init__()

        self.gnns = ModuleList( [GraphNet() for i in range(steps)] )
        self.pools = ModuleList( [GlobalNode() for i in range(steps)] )            
        # self.layer_norm = ModuleList( [LayerNorm(config.emb_dim) for i in range(steps)] )

        self.steps = steps

    def forward(self, x, xg_init, edge_attr, edge_index, batch_ind, num_graphs, data_lens):
        if xg_init is None:
            x_global = torch.zeros(num_graphs, config.emb_dim, device=config.device)
        else:
            x_global = x_init

        for i in range(self.steps):
            x = self.gnns[i](x, edge_attr, edge_index, x_global, batch_ind)            
            # x = self.layer_norm[i](x)

            x_global = self.pools[i](x_global, x, batch_ind)

        return x, x_global

# ----------------------------------------------------------------------------------------
class MultiMessagePassingWithGlobalNodeAndLastGRUGlobal(Module): # GRU in the global node
    def __init__(self, steps):
        super().__init__()

        self.gnns = ModuleList( [GraphNet() for i in range(steps)] )
        self.pools = ModuleList( [GlobalNode() for i in range(steps)] )            
        # self.layer_norm = ModuleList( [LayerNorm(config.emb_dim) for i in range(steps)] )

        self.gru = GRU(input_size=config.emb_dim, hidden_size=config.emb_dim)
        self.hidden = None

        self.steps = steps

    def forward(self, x, xg_init, edge_attr, edge_index, batch_ind, num_graphs, data_lens):
        if xg_init is None:
            x_global = torch.zeros(num_graphs, config.emb_dim, device=config.device)
        else:
            x_global = x_init

        # optionally initialize the hidden state
        if self.hidden is None:
            self.hidden = torch.zeros((1, num_graphs, config.emb_dim), device=config.device)

        for i in range(self.steps):
            x = self.gnns[i](x, edge_attr, edge_index, x_global, batch_ind)            
            # x = self.layer_norm[i](x)

            x_global = self.pools[i](x_global, x, batch_ind)

            if i == self.steps - 2: # pre-last
                x_global, self.hidden = self.gru(x_global.unsqueeze(0), self.hidden)
                x_global = x_global.squeeze(0)

        return x, x_global

    def reset_state(self, batch_mask):
        if self.hidden is not None:
            if batch_mask is None:  # reset all
                self.hidden = None

            else:
                # reset_idx = np.where(batch_mask)
                zeros = torch.zeros_like(self.hidden, device=config.device)
                condition = torch.zeros_like(self.hidden, dtype=torch.bool, device=config.device) # TODO: not optimal...
                condition[0, batch_mask] = True
                self.hidden = torch.where(condition, zeros, self.hidden) # reset the marked places

    # def detach_state(self):
    #     if self.hidden is not None:
    #         self.hidden = self.hidden.detach()

    def clone_state(self, other):
        if other.hidden is None:
            self.hidden = None

        else:
            self.hidden = other.hidden.clone().detach()

# ----------------------------------------------------------------------------------------
class MultiMessagePassingWithGlobalNodeAndAllGRUGlobal(Module): # GRU in the global node
    def __init__(self, steps):
        super().__init__()

        self.gnns = ModuleList( [GraphNet() for i in range(steps)] )
        self.pools = ModuleList( [GlobalNode() for i in range(steps)] )            
        # self.layer_norm = ModuleList( [LayerNorm(config.emb_dim) for i in range(steps)] )

        self.grus = ModuleList( [GRU(input_size=config.emb_dim, hidden_size=config.emb_dim) for i in range(steps)] )            
        self.hiddens = [None for i in range(steps)]
        
        # self.gru = GRU(input_size=config.emb_dim, hidden_size=config.emb_dim)
        # self.hidden = None

        self.steps = steps

    def forward(self, x, xg_init, edge_attr, edge_index, batch_ind, num_graphs, data_lens):
        if xg_init is None:
            x_global = torch.zeros(num_graphs, config.emb_dim, device=config.device)
        else:
            x_global = x_init

        # optionally initialize the hidden state
        if self.hiddens[0] is None:
            for i in range(self.steps):
                self.hiddens[i] = torch.zeros((1, num_graphs, config.emb_dim), device=config.device)

        for i in range(self.steps):
            x = self.gnns[i](x, edge_attr, edge_index, x_global, batch_ind)            
            # x = self.layer_norm[i](x)

            x_global = self.pools[i](x_global, x, batch_ind)

            x_global, self.hiddens[i] = self.grus[i](x_global.unsqueeze(0), self.hiddens[i])
            x_global = x_global.squeeze(0)


            # if i == self.steps - 2: # pre-last
            # if i == 0: # first
                # x_global, self.hidden = self.gru(x_global.unsqueeze(0), self.hidden)
                # x_global = x_global.squeeze(0)

        return x, x_global

    def reset_state(self, batch_mask):
        if self.hiddens[0] is not None:
            if batch_mask is None:  # reset all
                for i in range(self.steps):
                    self.hiddens[i] = None

            else:
                for i in range(self.steps):
                    # reset_idx = np.where(batch_mask)
                    zeros = torch.zeros_like(self.hiddens[i], device=config.device)
                    condition = torch.zeros_like(self.hiddens[i], dtype=torch.bool, device=config.device) # TODO: not optimal...
                    condition[0, batch_mask] = True
                    self.hiddens[i] = torch.where(condition, zeros, self.hiddens[i]) # reset the marked places

    # def detach_state(self):
    #     if self.hidden is not None:
    #         self.hidden = self.hidden.detach()

    def clone_state(self, other):
        if other.hiddens[0] is None:
            for i in range(self.steps):
                self.hiddens[i] = None

        else:
            for i in range(self.steps):
                self.hiddens[i] = other.hiddens[i].clone().detach()

# ----------------------------------------------------------------------------------------
class GlobalNode(Module):       
    def __init__(self):
        super().__init__()

        att_mask = Linear(config.emb_dim, 1)
        att_feat = Sequential( Linear(config.emb_dim, config.emb_dim), LeakyReLU() )

        self.glob = GlobalAttention(att_mask, att_feat)
        self.tranform = Sequential( Linear(config.emb_dim + config.emb_dim, config.emb_dim), LeakyReLU() )

    def forward(self, xg_prev, x, batch_ind):
        xg = self.glob(x, batch_ind)
        xg = torch.cat([xg, xg_prev], dim=1)
        xg = self.tranform(xg) + xg_prev # skip connection

        return xg

# ----------------------------------------------------------------------------------------
class GlobalPooling(Module):       
    def __init__(self):
        super().__init__()

        att_mask = Linear(config.emb_dim, 1)
        att_feat = Sequential( Linear(config.emb_dim, config.emb_dim), LeakyReLU() )

        self.glob = GlobalAttention(att_mask, att_feat)
        self.transform = Sequential( Linear(config.emb_dim, config.emb_dim), LeakyReLU() )

    def forward(self, x, batch_ind):
        x = self.glob(x, batch_ind)
        x = self.transform(x)

        return x

# ----------------------------------------------------------------------------------------
class GraphNet(MessagePassing):
    def __init__(self):
        super().__init__(aggr='max')

        # self.f_mess = Sequential( Linear(config.emb_dim + config.edge_dim, config.emb_dim), LeakyReLU(),
        #     Linear(config.emb_dim, config.emb_dim), LeakyReLU(),
        #     Linear(config.emb_dim, config.emb_dim), LeakyReLU())

        # self.f_agg  = Sequential( Linear(config.emb_dim + config.emb_dim + config.emb_dim, config.emb_dim), LeakyReLU(),
        #     Linear(config.emb_dim, config.emb_dim), LeakyReLU(),
        #     Linear(config.emb_dim, config.emb_dim), LeakyReLU())

        self.f_mess = Sequential( Linear(config.emb_dim + config.edge_dim, config.emb_dim), LeakyReLU() )
        self.f_agg  = Sequential( Linear(config.emb_dim + config.emb_dim + config.emb_dim, config.emb_dim), LeakyReLU() )

    def forward(self, x, edge_attr, edge_index, xg, batch_ind):
        xg = xg[batch_ind] # expand
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, xg=xg)

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            z = torch.cat([x_j, edge_attr], dim=1)
        else:
            z = x_j

        z = self.f_mess(z)

        return z 

    def update(self, aggr_out, x, xg):
        z = torch.cat([x, xg, aggr_out], dim=1)
        z = self.f_agg(z) + x # skip connection

        return z

# ------------------------------------------------------------------------------------------
class MultiSequential(Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)

        return inputs

# ------------------------------------------------------------------------------------------
class MultiGATv2(Module):
    def __init__(self, input_size, output_size, edge_size, layers):
        super().__init__()

        self.layers = []

        for l_id in range(layers):
            if l_id == 0: 
                in_size = input_size
            else:
                in_size = mid_size

            self.layers.append(MultiSequential(
                GATv2Conv(in_size, output_size),
                LeakyReLU()
            ))  

        self.layers = ModuleList(self.layers)

    def forward(self, x, edge_index, edge_attr):
        for l in self.layers:
            x = l(x, edge_index, edge_attr)

        return x

# ------------------------------------------------------------------------------------------
class MultiTransformer(Module):
    def __init__(self, input_size, output_size, edge_size, layers, heads, mean_at_end=False):
        super().__init__()

        self.layers = []
        mid_size = output_size * heads

        for l_id in range(layers):
            if l_id == 0: 
                in_size = input_size
            else:
                in_size = mid_size

            concat = True
            if l_id == layers - 1: # last
                concat = not mean_at_end

            self.layers.append(MultiSequential(
                TransformerConv(in_size, output_size, heads, concat=concat, beta=True, edge_dim=edge_size),
                LayerNorm(mid_size), 
                LeakyReLU()
            ))

        self.layers = ModuleList(self.layers)

    def forward(self, x, edge_index, edge_attr):
        for l in self.layers:
            x = l(x, edge_index, edge_attr)

        return x

# ------------------------------------------------------------------------------------------
def complete_matrix(data_lens):
    size = sum(data_lens)
    complete_adj = torch.zeros(size, size)
    
    start = 0
    for l in data_lens:
        complete_adj[start:start+l,start:start+l] = 1
        start += l

    # complete_adj -= torch.eye(size)  # remove self-connections
    return complete_adj.unsqueeze(0)

def complete_graph(data_lens):
    complete_adj = complete_matrix(data_lens)
    edge_index_complete, _ = torch_geometric.utils.dense_to_sparse(complete_adj)
    edge_index_complete = edge_index_complete.to(config.device)

    return edge_index_complete

# ------------------------------------------------------------------------------------------
class MultiAlternatingTransformer(Module):
    def __init__(self, input_size, output_size, edge_size, layers, heads, mean_at_end=False):
        super().__init__()

        self.layers = []
        mid_size = output_size * heads

        for l_id in range(layers):
            if l_id == 0: 
                in_size = input_size
            else:
                in_size = mid_size

            concat = True
            if l_id == layers - 1: # last
                concat = not mean_at_end

            edge_dim = edge_size if l_id % 2 == 0 else None

            self.layers.append(MultiSequential(
                TransformerConv(in_size, output_size, heads, concat=concat, beta=True, edge_dim=edge_dim),
                LayerNorm(mid_size), 
                LeakyReLU()
            ))

        self.layers = ModuleList(self.layers)

    def forward(self, x, edge_index, edge_attr, data_lens):
        # inefficient way
        edge_index_complete = complete_graph(data_lens)

        for l_id, l in enumerate(self.layers):
            if l_id % 2 == 0:
                x = l(x, edge_index, edge_attr)
            else:
                x = l(x, edge_index_complete, None)

        return x

# # ------------------------------------------------------------------------------------------
# from custom_lstm import LayerNormLSTMCell

# class MultiTransformerLSTM(Module):
#     def __init__(self, input_size, output_size, edge_size, layers, heads):
#         super().__init__()

#         self.layers = []
#         mid_size = output_size * heads

#         for i in range(layers):
#             if i == 0: 
#                 self.layers.append(
#                     MultiSequential(TransformerConv(input_size, output_size, heads, concat=True, beta=True, edge_dim=edge_size),
#                         LayerNorm(mid_size), LeakyReLU()))

#             else: 
#                 self.layers.append(
#                     MultiSequential(TransformerConv(mid_size, output_size, heads, concat=True, beta=True, edge_dim=edge_size),
#                     LayerNorm(mid_size), LeakyReLU()))

#         self.layers = ModuleList(self.layers)
#         self.lstm = LayerNormLSTMCell(input_size=mid_size, hidden_size=output_size)   # yes, it's reversed here

#         self.hc = None

#     def forward(self, x, edge_index, edge_attr):
#         for l in self.layers:
#             x = l(x, edge_index, edge_attr)

#         if self.hc is None:
#             self.hc = (torch.zeros(len(x), self.lstm.hidden_size, device=config.device), torch.zeros(len(x), self.lstm.hidden_size, device=config.device))

#         x, self.hc = self.lstm(x, self.hc)

#         return x

#     def reset_state(self, batch_mask):
#         if self.hc is not None:
#             if batch_mask is None:  # reset all
#                 self.hc = None

#             else:
#                 batch_mask = np.repeat(batch_mask, config.box_num_obj + 1)  # TODO: dynamically sized graphs?
#                 reset_idx = np.where(batch_mask)

#                 self.hc[0][reset_idx] = 0 # h_n, idx-batch-sample
#                 self.hc[1][reset_idx] = 0 # c_n

#     def detach_state(self):
#         if self.hc is not None:
#             self.hc = (self.hc[0].detach(), self.hc[1].detach())

#     def clone_state(self, other):
#         if other.hc is None:
#             self.hc = None

#         else:
#             self.hc = (other.hc[0].clone(), other.hc[1].clone())

if __name__ == '__main__':
    pos = torch.arange(4)
    pos_enc = positional_encoding(pos, dim=16)
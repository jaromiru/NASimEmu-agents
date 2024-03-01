import torch, numpy as np, random
from torch_geometric.data import Data, Batch

from nasimemu.nasim.envs.host_vector import HostVector

from config import config

A_ServiceScan = 0
A_OSScan = 1
A_SubnetScan = 2
A_ProcessScan = 3

class BaselineAgent():
    # static properties
    action_list = None
    exploit_list = None
    privesc_list = None

    def __init__(self):
        self.device = 'cpu'

    def prepare_batch(self, s_batch):
        node_data = [[HostVector(node) for node in scenario[:-1]] for scenario in s_batch]  # [:-1] = skip the last row which is for action result
        return node_data

    # find a suitable exploit
    def _get_exploit(self, node):
        n_services = [srv_id for srv_id, srv_val in node.services.items() if srv_val == 1]
        n_os = [os_id for os_id, os_val in node.os.items() if os_val == 1][0]

        for e_key, e_val in self.exploit_list:
            if e_val['os'] == n_os and e_val['service'] in n_services:
                return e_key

    # find a suitable privesc
    def _get_privesc(self, node):
        n_procs = [proc_id for proc_id, proc_val in node.processes.items() if proc_val == 1]
        n_os = [os_id for os_id, os_val in node.os.items() if os_val == 1][0]
            
        for pe_key, pe_val in self.privesc_list:
            if pe_val['os'] == n_os and pe_val['process'] in n_procs:
                return pe_key

    # find a suitable host to exploit
    def _exploit_hosts(self, s_id, nodes, nodes_id):
        # exploit hosts
        candidates = [(node, self._get_exploit(node)) for node in nodes if 
                        node.access == 0 and
                        node.address in self.scanned_services[s_id] and
                        node.address in self.scanned_oss[s_id]]
        
        # debug
        # for n, n_e in candidates:
        #     if n_e is None:
        #         print(f"No exploit found for node {n}.")

        candidates = [x for x in candidates if x[1] is not None]

        if len(candidates) > 0:
            node, exploit = random.choice(candidates)
            action_id = self.action_list.index(exploit)

            return (node.address, action_id)
        
    # privesc sensitive hosts with user access
    def _privesc(self, s_id, nodes, nodes_id):
        candidates = [(node, self._get_privesc(node)) for node in nodes if node.access == 1 and node.value > 0 and node.address in self.scanned_procs[s_id]]

        # debug
        # for n, n_pe in candidates:
        #     if n_pe is None:
        #         print(f"No privesc found for node {n}.")

        candidates = [x for x in candidates if x[1] is not None]
        if len(candidates) > 0:
            node, privesc = random.choice(candidates)
            action_id = self.action_list.index(privesc)

            return (node.address, action_id)

    # scan processes
    def _scan_processes(self, s_id, nodes, nodes_id):
        candidates = [node for node in nodes if node.access == 1 and node.value > 0 and node.address not in self.scanned_procs[s_id]]
        if len(candidates) > 0:
            node = random.choice(candidates)
            self.scanned_procs[s_id].add(node.address)
            return (node.address, A_ProcessScan)    
        
    # scan subnets, we need an exploited node in unscanned subnet
    def _scan_subnets(self, s_id, nodes, nodes_id):
        candidates = [node for node in nodes if node.access >= 1 and node.address[0] not in self.scanned_subnets[s_id]]
        if len(candidates) > 0:
            node = random.choice(candidates)
            self.scanned_subnets[s_id].add(node.address[0])
            return (node.address, A_SubnetScan)

    # scan host services
    def _scan_services(self, s_id, nodes, nodes_id):
        candidates = list(nodes_id - self.scanned_services[s_id])

        if len(candidates) > 0:
            node = random.choice(candidates)
            self.scanned_services[s_id].add(node)
            return (node, A_ServiceScan)  

    # scan host os
    def _scan_os(self, s_id, nodes, nodes_id):
        candidates = list(nodes_id - self.scanned_oss[s_id])
        if len(candidates) > 0:
            node = random.choice(candidates)
            self.scanned_oss[s_id].add(node)
            return (node, A_OSScan)           

    def _select_action(self, s_id, nodes):
        nodes_id = {x.address for x in nodes}

        pipeline = [
            self._privesc,
            self._scan_processes,
            self._exploit_hosts,
            self._scan_services,
            self._scan_os,
            self._scan_subnets
        ]

        for f in pipeline:
            res = f(s_id, nodes, nodes_id)
            if res is not None:
                return res

        # no other action possible: issue the terminal action
        return (None, -1)

    def __call__(self, s_batch):
        node_data = self.prepare_batch(s_batch)

        if self.init_req:
            self.init_req = False   
            
            self.scanned_services = [set() for x in range(len(node_data))]
            self.scanned_procs = [set() for x in range(len(node_data))]
            self.scanned_oss = [set() for x in range(len(node_data))]
            self.scanned_subnets = [set() for x in range(len(node_data))]

        actions = []
        for s_id, nodes in enumerate(node_data):
            actions.append( self._select_action(s_id, nodes) )

        return actions, None, None, None
            
    def get_param_count(self):
        return 0

    def clone_state(self, o):
        pass

    def reset_state(self, batch_mask=None):
        if batch_mask is not None:
            for idx, reset in enumerate(batch_mask):
                if reset:
                    self.scanned_services[idx] = set()
                    self.scanned_procs[idx] = set()
                    self.scanned_oss[idx] = set()
                    self.scanned_subnets[idx] = set()

        else:
            self.init_req = True

    def eval(self):
        pass

    def train(self):
        pass
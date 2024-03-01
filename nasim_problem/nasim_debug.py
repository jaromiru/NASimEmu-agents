from vec_env.subproc_vec_env import SubprocVecEnv
from tqdm import tqdm
import gym, numpy as np, torch
import itertools

from config import config

from nasimemu.nasim.envs.action import Exploit, PrivilegeEscalation, ServiceScan, OSScan, SubnetScan, ProcessScan
from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.env_utils import convert_to_graph, plot_network

import plotly.io as pio  

ACTION_MAP = {
    ServiceScan: 'ServScan',
    OSScan: 'OSScan',
    SubnetScan: 'SubnetScan',
    ProcessScan: 'ProcScan',
    Exploit: 'Exploit',
    PrivilegeEscalation: 'PrivEsc'
}

class NASimDebug():
    def calc_baseline(self):
        # calculate the mean success try for probability p
        def get_expected_success(p, l=100):
            assert p >= 0.1, 'Computation not precise for p < 0.1; increase l'

            n_0 = np.arange(l)
            n_1 = n_0 + 1

            p_not = np.full(l, 1-p) ** n_0
            p_mul = p_not * p

            p_mean = np.sum( p_mul * n_1 )
            return p_mean

        test_env = gym.make('NASimEmuEnv-v99', random_init=False)
        env_raw = test_env.env
        test_env.reset()

        # useful constants
        p_e = np.mean([e_data['prob'] for e_name, e_data in env_raw.exploit_list])
        p_pe = np.mean([pe_data['prob'] for pe_name, pe_data in env_raw.privesc_list])

        exp_e = get_expected_success(p_e) # we assume exploit success probability = 0.8
        exp_pe = get_expected_success(p_pe) # 

        # stats
        req_actions_list = []
        reward_list = []

        reward_per_action = -0.1
        reward_per_sensitive_host = 10.

        sensitive_hosts = 0

        # need to iterate and average, because there is randomness in the generated scenario
        for i in range(config.eval_problems):
            s = test_env.reset()
            network = env_raw.env.network

            req_actions = ( len(network.hosts) +                          # service scan each host
                            exp_e * len(network.hosts) +                  # exploit each host   
                            exp_pe * len(network.get_sensitive_hosts()) + # privesc each sensitive host
                            len(network.subnets) )                        # scan in each of the subnets once to discover whole network

            if len(env_raw.env.scenario.privescs) > 1:            # only if there are multiple privescs
                req_actions += len(network.get_sensitive_hosts()) # process scan for each sensitive host

            reward = reward_per_action * req_actions + \
                     reward_per_sensitive_host * len(network.get_sensitive_hosts())

            sensitive_hosts += len(network.get_sensitive_hosts())

            req_actions_list.append(req_actions)
            reward_list.append(reward)

        req_actions_mean = np.mean(req_actions_list)
        reward_mean = np.mean(reward_list)

        print(f"This scenario contains {sensitive_hosts / config.eval_problems} sensitive hosts on average.")
        print(f"{test_env.env.scenario_name};{reward_mean:.2f};{req_actions_mean:.2f}")

    def evaluate(self, net):
        log_trn = self._eval(net, config.scenario_name)

        if config.test_scenario_name:
            log_tst = self._eval(net, config.test_scenario_name)
        else:
            log_tst = None

        return {'eval_trn': log_trn, 'eval_tst': log_tst}

    def _eval(self, net, scenario_name):
        test_env = SubprocVecEnv([lambda: gym.make('NASimEmuEnv-v99', random_init=False, scenario_name=scenario_name) for i in range(config.eval_batch)], in_series=(config.eval_batch // config.cpus), context='fork')
        tqdm_val = tqdm(desc='Validating', unit=' problems')

        saved_state = net.__class__() # create a fresh instance
        saved_state.clone_state(net)
        
        with torch.no_grad():
            net.eval()
            net.reset_state()

            r_tot = 0.
            r_episodes = 0.
            problems_solved = 0
            problems_finished = 0
            episode_lengths = 0
            captured = 0
            steps = 0
            terminated = np.zeros(config.eval_batch, dtype=bool)

            s = test_env.reset()

            while True:
                steps += np.sum(~terminated)

                a, v, pi, _ = net(s)
                a = np.array(a, dtype=object)

                s, r_, d_, i_ = test_env.step(a)
                net.reset_state(d_)

                r = r_[~terminated]
                d = d_[~terminated]
                i = list(itertools.compress(i_, ~terminated))

                # print(r)
                r_tot += np.sum(r)
                r_episodes += sum(x['r_tot'] for x in i if x['done'] == True)

                problems_solved   += sum('d_true' in x and x['d_true'] == True for x in i)
                problems_finished += np.sum(d)
                
                episode_lengths += sum(x['step_idx'] for x in i if x['done'] == True)
                captured += sum(x['captured'] for x in i if x['done'] == True)

                tqdm_val.update(np.sum(d))

                if problems_finished >= config.eval_problems:
                    terminated |= d_
                    if np.all(terminated):
                        break

            r_avg = r_tot / steps # average reward per step <- obsolete! this is a wrong metric to track in case there are differences in episode lengths
            r_avg_episodes = r_episodes / problems_finished

            problems_solved_ps  = problems_solved / steps
            problems_solved_avg = problems_solved / problems_finished

            episode_lengths_avg = episode_lengths / problems_finished
            captured_avg = captured / problems_finished

            net.train()

        net.clone_state(saved_state)

        tqdm_val.close()
        test_env.close()

        log = {
            'reward_avg': r_avg,
            'reward_avg_episodes': r_avg_episodes,
            'eplen_avg': episode_lengths_avg,
            'captured_avg': captured_avg
            # 'solved_per_step': problems_solved_ps,
            # 'solved_avg': problems_solved_avg,
        }

        return log

    def debug(self, net, show=False):
        test_env = gym.make('NASimEmuEnv-v99', random_init=False)
        s = test_env.reset()
        
        saved_state = net.__class__() # create a fresh instance
        saved_state.clone_state(net)
        
        with torch.no_grad():
            net.eval()
            net.reset_state()
            node_softmax, action_softmax, value, q_val = net([s], complete=True)
            net.train()

        net.clone_state(saved_state)

        G = self._make_graph(s, node_softmax, action_softmax)
        fig = self._plot(G, value.item(), q_val.item(), test_env)

        if show:
            fig.show()

        log = {
            'value': value, 
            'q_val': q_val,
            'figure': fig
        }

        return log

    def trace(self, net, net_name):
        with torch.no_grad():
            test_env = gym.make('NASimEmuEnv-v99', verbose=False, random_init=False, emulate=config.emulate)
            s = test_env.reset()
            i = {'subnet_graph': test_env.subnet_graph.copy()}

            # self._plot_network(test_env, net, s)

            net.reset_state()

            if not config.emulate:
                print("Note: This is simulation state - it is not observed.")
                test_env.env.env.render_state()
            else:
                print("Note: Emulation - true state unknown.")

            test_env.env.env.render(obs=test_env.s_raw)
            # input()
        
            pio.kaleido.scope.mathjax = None

            for step in range(1, 200):
                print(f"\nSTEP {step}")
                a, v, pi, _ = net([s])
                # a = np.array(a, dtype=object) # todo: test when error occurs

                s_raw_orig = test_env.s_raw
                orig_subnet_graph = i['subnet_graph'].copy()

                s, r, d, i = test_env.step(a[0])
                net.reset_state([d])

                print()
                print(f"a: {i['a_raw']}, r: {r}, d: {d}")
                # print(f"V(s)={v.item():.2f}, Q(s, a_cnt)={q.item():.2f}")
                if v is not None:
                    print(f"V(s)={v.item():.2f}")

                fig = plot_network(s_raw_orig, orig_subnet_graph, i['a_raw'])
                # fig.show()

                fig.update_xaxes(range=[-1.35, 1.3])
                fig.update_yaxes(range=[-1.2, 1.3])
                fig.write_image(f"out/trace-{step}.pdf", width=1200, height=600, scale=1.0)

                if i['success']:
                    test_env.env.env.render(obs=i['s_raw'])

                if d:
                    print("-------------FINISHED----------------")
                    exit()

                    # input()

    # # taken from https://plotly.com/python/network-graphs/
    # def _plot(self, G, value, q_val, env):
    #     edge_x = []
    #     edge_y = []
    #     for edge in G.edges():
    #         x0, y0 = G.nodes[edge[0]]['pos']
    #         x1, y1 = G.nodes[edge[1]]['pos']
    #         edge_x.append(x0)
    #         edge_x.append(x1)
    #         edge_x.append(None)
    #         edge_y.append(y0)
    #         edge_y.append(y1)
    #         edge_y.append(None)

    #     edge_trace = go.Scatter(
    #         x=edge_x, y=edge_y,
    #         line=dict(width=0.5, color='#444'),
    #         hoverinfo='none',
    #         mode='lines')

    #     node_x = []
    #     node_y = []
    #     node_text = []
    #     node_color = []
    #     # node_width = []
    #     # node_probs = []
    #     for node_id, node in G.nodes.items():
    #         x, y = node['pos']
    #         node_x.append(x)
    #         node_y.append(y)
    #         # node_text.append(f"{node['label']}: {node['n_prob']:.2f} / {np.array_str(node['a_prob'], precision=2)}")
            
    #         # action_class, action_params = env.action_list[np.argmax(node['a_prob'])]
    #         # action_name = action_params.get('name', '')

    #         # if node['type'] == 'node':
    #         #   node_text.append(f"{node['label']}: {node['n_prob']:.2f} / {action_class.__name__} {action_name}")
    #         # else:
    #         node_text.append(f"{node['label']}")

    #         # node_probs.append(np.array_str(node['a_prob'], precision=2))

    #         node_color.append(node['color'])
    #         # node_width.append(node['n_prob'] * 10.)

    #     node_trace = go.Scatter(
    #         x=node_x, y=node_y,
    #         mode='markers+text',
    #         hoverinfo='text',
    #         marker=dict(showscale=False, color=node_color, size=15, line_width=1.0),
    #         text=node_text,
    #         textposition="top center",)
    #         # customdata=node_probs,
    #         # hovertemplate="%{customdata}")

    #     fig = go.Figure(data=[edge_trace, node_trace],
    #                  layout=go.Layout(
    #                     showlegend=False,
    #                     hovermode='closest',
    #                     margin=dict(b=0,l=0,r=0,t=0),
    #                     # annotations=[dict(text=f"V(s)={value:.2f}, Q(s, a_cnt)={q_val:.2f}",
    #                     # annotations=[dict(text=f"V(s)={value:.2f}",
    #                     #     showarrow=False,
    #                     #     xref="paper", yref="paper",
    #                     #     x=0.005, y=-0.002)],
    #                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    #                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    #                     )

    #     fig.update_layout(
    #         paper_bgcolor="rgba(0,0,0,0)",
    #         plot_bgcolor="rgba(0,0,0,0)"
    #     )

    #     return fig

    # def _plot_new(self, graph, node_data):
    #     edge_x = []
    #     edge_y = []
    #     for edge in graph.edges():
    #         x0, y0 = node_data['x'][edge[0]], node_data['y'][edge[0]]
    #         x1, y1 = node_data['x'][edge[1]], node_data['y'][edge[1]]
    #         edge_x.append(x0)
    #         edge_x.append(x1)
    #         edge_x.append(None)
    #         edge_y.append(y0)
    #         edge_y.append(y1)
    #         edge_y.append(None)

    #         print(edge)
    #         print(x0, y0, x1, y1)

    #     edge_trace = go.Scatter(
    #         x=edge_x, y=edge_y,
    #         line=dict(width=0.5, color='#444'),
    #         hoverinfo='none',
    #         mode='lines')

    #     node_trace = go.Scatter(
    #         node_data,
    #         mode='markers+text',
    #         # hoverinfo='text',
    #         # marker=dict(showscale=False, size=15,),
    #         textposition="top center")

    #     fig = go.Figure(data=[edge_trace, node_trace],
    #                     layout=go.Layout(
    #                         showlegend=False,
    #                         hovermode='closest',
    #                         margin=dict(b=0,l=0,r=0,t=0),
    #                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    #                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)),
    #                     )

    #     fig.update_layout(
    #         paper_bgcolor="rgba(0,0,0,0)",
    #         plot_bgcolor="rgba(0,0,0,0)"
    #     )

    #     return fig


    # def _make_graph_new(self, s):
    #     node_feats, edge_index, node_index = s

    #     graph = nx.Graph() 
    #     graph.add_edges_from(edge_index.T) 
    #     node_positions = nx.kamada_kawai_layout(graph)
    #     print(graph.nodes)
    #     print(node_positions)
    #     node_positions = np.stack([(node_positions[i]) for i in range(len(graph.nodes))])

    #     node_data = {}
    #     node_data['x'] = node_positions[:, 0]
    #     node_data['y'] = node_positions[:, 1]

    #     node_color = lambda node_id: f'rgb({scenario["sensitive_hosts"][node_id] * 191 + 64}, 64, 64)'

    #     node_data['text']  = [f"Subnet {node_index[i][0]}" if node_index[i][1] == -1 else f"{node_index[i]}" for i in graph.nodes]
    #     node_data['marker'] = dict(opacity=1.0, size=15, color=['seagreen' if node_index[i][1] == -1 else 'skyblue' for i in graph.nodes])

    #     return graph, node_data

from nasimemu.env import NASimEmuEnv

from .nasim_net_mlp import NASimNetMLP # Multi-layer perceptron
from .nasim_net_mlp_lstm import NASimNetMLP_LSTM # Multi-layer perceptron + LSTM

from .nasim_net_inv import NASimNetInv # Permutation invariant + compound action (p_node * p_act)
from .nasim_net_inv_mact import NASimNetInvMAct # Inv + Matrix action
from .nasim_net_inv_mact_train_at import NASimNetInvMActTrainAT # Inv + Matrix action + trainable a_t
from .nasim_net_inv_mact_lstm import NASimNetInvMActLSTM # Inv + Matrix action + GRU

from .nasim_net_gnn import NASimNetGNN	# Graph NN
from .nasim_net_gnn_mact import NASimNetGNN_MAct # Graph NN + Matrix action
from .nasim_net_gnn_lstm import NASimNetGNN_LSTM # Graph NN with LSTM

from .nasim_net_xatt import NASimNetXAtt # Attention
from .nasim_net_xatt_mact import NASimNetXAttMAct # Attention 

from .nasim_baseline import BaselineAgent 

class NASimConfig():
	@staticmethod
	def update_config(config, args):
		config.emulate = args.emulate

		# config.scenario_name = 'medium-gen-rgoal'
		# config.scenario_name = "nasim/scenarios/benchmark/tiny.yaml"
		# config.scenario_name = "nasim_emulation/scenarios/simple_03.yaml"
		config.scenario_name = args.scenario
		config.test_scenario_name = args.test_scenario

		# config.node_dim = 34
		# config.step_limit = 200
		config.step_limit = args.episode_step_limit
		config.use_a_t = args.use_a_t

		# config.scenario_name = 'huge-gen-rgoal'
		# config.node_dim = 43
		# config.step_limit = 400

		config.edge_dim = 0
		config.pos_enc_dim = 8

		config.fully_obs = args.fully_obs
		config.observation_format = args.observation_format
		config.augment_with_action = args.augment_with_action

		config.net_class = eval(args.net_class)
		
		# config.net_class = NASimNetXAtt
		# config.net_class = NASimNetMLP
		# config.net_class = NASimNetInv
		# config.net_class = NASimNetInvMAct
		# config.net_class = NASimNetGNN
		# config.net_class = NASimNetGNN_LSTM

		# calculate number of actions
		env = NASimEmuEnv(scenario_name=config.scenario_name, augment_with_action=config.augment_with_action)
		s = env.reset()

		config.action_dim = len(env.action_list)
		config.node_dim = s.shape[1] + 1 # + 1 feature (node/subnet)

		if config.net_class == BaselineAgent:
			BaselineAgent.action_list = [x[1]['name'] if 'name' in x[1] else None for x in env.action_list]	 # action ids

			BaselineAgent.exploit_list = env.exploit_list
			BaselineAgent.privesc_list = env.privesc_list
			
		# Exploit
		# PrivilegeEscalation
		# ServiceScan
		# OSScan
		# SubnetScan
		# ProcessScan

	@staticmethod
	def update_argparse(argparse):
		argparse.add_argument('scenario', type=str, help="Path to scenario to load. You can specify multiple scenarios with ':', just make sure that they share the same 'address_space_bounds'.")
		argparse.add_argument('-fully_obs', action='store_const', const=True, help="Use fully observable environment (default: False)")

		argparse.add_argument('-observation_format', type=str, default='list', help="list / graph")
		argparse.add_argument('-augment_with_action', action='store_const', const=True, help="Include the last action in observation (useful with LSTM)")
		argparse.add_argument('-net_class', type=str, default='NASimNetMLP', choices=['BaselineAgent', 'NASimNetMLP', 'NASimNetMLP_LSTM', 'NASimNetInv', 'NASimNetInvMAct', 'NASimNetInvMActTrainAT', 'NASimNetInvMActLSTM', 'NASimNetGNN', 'NASimNetGNN_MAct', 'NASimNetGNN_LSTM', 'NASimNetXAtt', 'NASimNetXAttMAct'])

		argparse.add_argument('-episode_step_limit', type=int, default=200, help="Force termination after number of steps")
		argparse.add_argument('-use_a_t', action='store_const', const=True, help="Enable agent to terminate the episode")

		argparse.add_argument('--emulate', action='store_const', const=True, help="Emulate the network (via vagrant; use only with --trace)")
		argparse.add_argument('--test_scenario', type=str, help="Additional test scenarios to separately test the model (aka train/test datasets).")

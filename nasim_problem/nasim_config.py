from nasimemu.env import NASimEmuEnv

from .nasim_net_inv_mact import NASimNetInvMAct # Inv + Matrix action
from .nasim_net_mlp import NASimNetMLP # Multi-layer perceptron

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
		config.step_limit = 20

		# config.scenario_name = 'huge-gen-rgoal'
		# config.node_dim = 43
		# config.step_limit = 400

		config.edge_dim = 0

		config.fully_obs = args.fully_obs
		# config.observation_format = 'graph'
		config.observation_format = 'list'
		config.net_class = NASimNetMLP
		# config.net_class = NASimNetInvMAct

		# calculate number of actions
		env = NASimEmuEnv(scenario_name=config.scenario_name)
		env.reset()

		config.action_dim = len(env.action_list)
		config.node_dim = env.env.current_state.tensor.shape[1] + 1 # + 1 feature if its node/subnet

		# Exploit
		# PrivilegeEscalation
		# ServiceScan
		# OSScan
		# SubnetScan
		# ProcessScan

	@staticmethod
	def update_argparse(argparse):
		argparse.add_argument('-fully_obs', action='store_const', const=True, help="Use fully observable environment (default: False)")
		argparse.add_argument('--emulate', action='store_const', const=True, help="Emulate the network (via vagrant; use only with --trace)")
		argparse.add_argument('scenario', type=str, help="Path to scenario to load. You can specify multiple scenarios with ':', just make sure that they share the same 'address_space_bounds'.")
		argparse.add_argument('--test_scenario', type=str, help="Additional test scenarios to separately test the model (aka train/test datasets).")

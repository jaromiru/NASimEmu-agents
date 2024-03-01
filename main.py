import numpy as np
import gym, torch, logging

import wandb, argparse, itertools, os, random

from vec_env.subproc_vec_env import SubprocVecEnv
from tqdm import tqdm

from config import config
from nasim_problem import NASimRRL as Problem

# ----------------------------------------------------------------------------------------
def decay_time(step, start, min, factor, rate):
	exp = step / rate * factor
	value = (start - min) / (1 + exp) + min

	return value

def decay_exp(step, start, min, factor, rate):
	exp = step / rate
	value = (start - min) * (factor ** exp) + min

	return value

def init_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

def get_args(problem_config):
	cuda_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]

	# optimal cpu=2, device=cuda (rate 3.5)
	parser = argparse.ArgumentParser()
	parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'] + cuda_devices, default='cpu', help="Which device to use")
	parser.add_argument('-cpus', type=str, default='2', help="How many CPUs to use")
	parser.add_argument('-batch', type=int, default=128, help="Number of parallel environments")
	parser.add_argument('-seed', type=int, default=None, help="Random seed") # seed in multiprocessing is not implemented
	parser.add_argument('-load_model', type=str, default=None, help="Load model from this file")
	parser.add_argument('-epoch', type=int, default=1000, help="Epoch length")
	parser.add_argument('-max_epochs', type=int, default=None, help="Terminate after this many epochs")

	parser.add_argument('-mp_iterations', type=int, default=3, help="Number of message passes")
	parser.add_argument('-emb_dim', type=int, default=64, help="Embedding size")

	parser.add_argument('-force_continue_epochs', type=int, default=0, help="Disable force continue after this epochs (0=disable immediately; -1=never disable)")

	parser.add_argument('-lr', type=float, default=3e-3, help="Initial learning rate")
	parser.add_argument('-alpha_h', type=float, default=0.3, help="Initial entropy regularization constant")
	parser.add_argument('-max_norm', type=float, default=3., help="Maximal gradient norm")

	parser.add_argument('--trace', action='store_const', const=True, help="Show trace of the agent")
	parser.add_argument('--eval', action='store_const', const=True, help="Evaluate the agent")
	parser.add_argument('--debug', action='store_const', const=True, help="Debug the agent")
	parser.add_argument('--calc_baseline', action='store_const', const=True, help="Calculate required steps of a baseline agent")
	
	parser.add_argument('--no_debug', action='store_const', const=True, help="Do not debug the agent")

	# delegate argparse to problem config
	problem_config.update_argparse(parser)

	cmd_args = parser.parse_args()
	return cmd_args

# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
	# logging.basicConfig(level=logging.INFO)
	logging.basicConfig(level=logging.DEBUG)
	logging.getLogger('urllib3').setLevel(logging.INFO)
	logging.getLogger('numba').setLevel(logging.INFO)

	problem = Problem()
	problem_config = problem.make_config()

	np.set_printoptions(threshold=np.inf, precision=4, suppress=True)

	args = get_args(problem_config)
	config.init(args)
	problem_config.update_config(config, args) # update config with problem specific settings

	print(f"Config: {config}")

	if config.seed:
		init_seed(config.seed)

	torch.set_num_threads(config.cpus)	

	problem.register_gym()
	problem_debug = problem.make_debug()
	
	if args.calc_baseline:
		problem_debug.calc_baseline()
		exit(0)

	net = problem.make_net()
	target_net = problem.make_net()
	print(net)
	print(f"Number of parameters: {net.get_param_count()}")

	if config.load_model:
		net.load(config.load_model)
		target_net.load(config.load_model)

		print(f"Model loaded: {config.load_model}")

	if args.trace:
		problem_debug.trace(net, config.load_model)
		exit(0)

	if args.eval:
		import pprint
		eval_res = problem_debug.evaluate(net)
		# print(f"Avg. reward: {r_avg}, Avg. solved per step: {s_ps_avg}, Avg. solved: {s_avg}, Tot. finished: {s_tot}")
		pprint.pp(eval_res)
		exit(0)

	if args.debug:
		problem_debug.debug(net, show=True)
		exit(0)

	env = SubprocVecEnv([lambda: gym.make(problem.get_gym_name()) for i in range(config.batch)], in_series=(config.batch // config.cpus), context='fork')

	wandb.init(project=problem.get_project_name(), name=problem.get_run_name(), config=config)
	wandb.watch(net, log='all')

	tot_env_steps = 0
	norm_log = []
	entropy_log = []

	if config.force_continue_steps == 0:
		print("Disabling force_continue")
		net.set_force_continue(False)
	else:
		print("Enabling force_continue")
		net.set_force_continue(True)

	tqdm_main = tqdm(desc='Training', unit=' steps')
	s = env.reset()

	for step in itertools.count(start=1):
		trace = []

		hidden_s0 = problem.make_net()		# save internal (recurrent) network state at s_0 and s_last
		hidden_s0.clone_state(net)

		for step_trace in range(config.ppo_t):
			s_orig = s

			a, v, pi, raw_a = net(s)
			a = np.array(a, dtype=object)
			s, r, d, i = env.step(a)
			net.reset_state(d)

			a_cnt = [0 if a_action == -1 else 1 for (a_node, a_action) in a] # action_q - 0 = terminate / 1 = continue

			s_true = [x['s_true'] for x in i]
			d_true = [x['d_true'] for x in i] # note: currently d == d_true (dependency in v_target, q_target computations and reccurency in ppo

			trace.append( (s_orig, raw_a, a_cnt, r, s_true, d_true) )

		# update network
		# loss, entropy, norm, pi_deviations = net.update(s_orig, raw_a, r, s_true, d_true, target_net)
		target_net.clone_state(net)
		loss, entropy, norm, pi_deviations = net.update(trace, target_net, hidden_s0)
		target_net.copy_weights(net, rho=config.target_rho)

		# print([x.item() for x in pi_deviations])

		# save step stats
		tot_env_steps += config.batch
		tqdm_main.update()

		norm_log.append(norm)
		entropy_log.append(entropy)

		if step % config.sched_lr_rate == 0:
			lr = decay_exp(step, config.opt_lr, config.sched_lr_min, config.sched_lr_factor, config.sched_lr_rate)
			net.set_lr(lr)

		if step % config.sched_alpha_h_rate == 0:
			alpha_h = decay_time(step, config.alpha_h, config.sched_alpha_h_min, config.sched_alpha_h_factor, config.sched_alpha_h_rate)
			net.set_alpha_h(alpha_h)

		if step % config.log_rate == 0:
			log_step = step // config.log_rate

			# r_avg, s_ps_avg, s_avg, _ = problem_debug.evaluate(net)
			# r_avg_trn, s_ps_avg_trn, s_avg_trn, _ = problem_debug.evaluate(net, split='train', subset=config.subset)

			eval_perf = problem_debug.evaluate(net)
			# log_trn_eval = problem_debug.evaluate(net, split='train', subset=config.subset)

			if args.no_debug:
				log_debug = None
			else:
				log_debug = problem_debug.debug(net)
				# print(log_debug['value'], log_debug['q_val'])
		
			log = {
				'env_steps': tot_env_steps,
				# 'el_env_steps': tot_el_env_steps,
				'rate': tqdm_main.format_dict['rate'],

				'loss': loss,
				# 'loss_pi': loss_pi,
				# 'loss_v': loss_v,
				# 'loss_h': loss_h,

				'pi_deviations': wandb.Histogram(pi_deviations),

				'grad_mean': np.mean(norm_log),
				'grad_min': np.min(norm_log),
				'grad_max': np.max(norm_log),

				'entropy_mean': np.mean(entropy_log),
				'entropy_min': np.min(entropy_log),
				'entropy_max': np.max(entropy_log),

				'lr': net.lr,
				'alpha_h': net.alpha_h,

				'eval_perf': eval_perf,

				'debug': log_debug,
			}

			norm_log = []
			entropy_log = []

			print(log)
			wandb.log(log)

			# save model to wandb
			model_file = os.path.join(wandb.run.dir, "model.pt")
			net.save(model_file)
			wandb.save(model_file)

		# finish if max_epochs exceeded
		if config.max_epochs and (step // config.log_rate >= config.max_epochs):
			break

		if step == config.force_continue_steps:
			print("Disabling force_continue")
			net.set_force_continue(False)

	env.close()
	tqdm_main.close()
import torch, os, numpy as np

def get_cpu_count(cpus):
	if cpus == 'auto':
		slurm_cpus = os.environ.get('SLURM_CPUS_ON_NODE')
		node_cpus = os.cpu_count()

		if slurm_cpus is not None:
			return int(slurm_cpus)
		else:
			return node_cpus

	else:
		return int(cpus)

def get_device(device):
	if device == 'auto':
		return 'cuda' if torch.cuda.is_available() else 'cpu'
	else:
		return device

class Object:
	def init(self, args):
		self.gamma = 0.99
		self.batch = args.batch

		self.epoch = args.epoch

		self.ppo_k = 3
		self.ppo_t = 8
		self.ppo_eps = 0.2

		self.alpha_v = 0.1 / self.ppo_k
		self.alpha_h = args.alpha_h

		self.force_continue_steps = args.force_continue_epochs * self.epoch

		self.target_rho = 0.1
		self.emb_dim = args.emb_dim
		self.mp_iterations = args.mp_iterations

		self.seed = args.seed
		self.device = get_device(args.device)
		self.cpus = min(get_cpu_count(args.cpus), args.batch)

		self.opt_lr = args.lr
		self.opt_l2 = 1.0e-4
		self.opt_max_norm = args.max_norm

		self.sched_lr_factor = 0.5
		self.sched_lr_min    = self.opt_lr / 30
		self.sched_lr_rate   = 25 * self.epoch
		
		self.sched_alpha_h_factor = 1.0
		self.sched_alpha_h_min    = self.alpha_h / 100.
		self.sched_alpha_h_rate   = 10 * self.epoch

		self.v_range = None #(-np.inf, np.inf)	# range of the value function to help the optimization, can be None

		self.max_epochs = args.max_epochs
		self.log_rate = 1 * self.epoch
		self.eval_problems = 100
		self.eval_batch = 64

		self.load_model = args.load_model

		self.slurm_job_id = os.environ.get('SLURM_JOB_ID')

	def __str__(self):
		return str( vars(self) )

config = Object()



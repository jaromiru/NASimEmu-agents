from nasimemu.env import NASimEmuEnv

from .nasim_debug import NASimDebug
from .nasim_config import NASimConfig

from config import config
import gym

class NASimRRL():
	@staticmethod
	def make_env():
		return NASimEmuEnv()

	@staticmethod
	def make_net():
		return config.net_class()

	@staticmethod
	def make_debug():
		return NASimDebug()

	@staticmethod
	def make_config():
		return NASimConfig()

	@staticmethod
	def register_gym():
		gym.envs.registration.register(
			id='NASimEmuEnv-v99',
			entry_point='nasimemu.env:NASimEmuEnv',
			kwargs={'scenario_name': config.scenario_name, 'step_limit': config.step_limit, 
				'fully_obs': config.fully_obs, 'observation_format': config.observation_format,
				'augment_with_action': config.augment_with_action, 'random_init': True, 'verbose': False}
		)

	@staticmethod
	def get_gym_name():
		return 'NASimEmuEnv-v99'

	@staticmethod
	def get_project_name():
		return 'rrl-nasim'

	@staticmethod
	def get_run_name():
		return None

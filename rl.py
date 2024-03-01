import torch, numpy as np

# reward(s,a), value(s), value(s_), pi(a|s)
def a2c(r, v, v_, pi, gamma, alpha_v, alpha_h, q_range=None, num_actions=None):
	# make sure all dimensions are correct
	r = r.flatten()
	v = v.flatten()
	v_ = v_.flatten()
	pi = pi.flatten()
	num_actions = num_actions.flatten()

	# compute losses
	log_pi = torch.log(pi + 1e-10)	# bug fix in torch.multinomial: this should never be zero... (actually, the product may)
	q = r + gamma * v_.detach()
	v_target = q if q_range is None else q.clamp(*q_range)

	adv = q - v
	v_err = v_target - v

	loss_pi = -adv.detach() * log_pi
	loss_v  = v_err ** 2					# can use experience replay here

	if num_actions is not None:
		legal_actions = num_actions > 1
		num_actions[~legal_actions] = 2	# bug fix in pytorch: to avoid nan error during backprop
		loss_h = (log_pi.detach() * log_pi) / torch.log(num_actions) # scale the entropy with its maximum

		loss_h = loss_h[legal_actions]

		ent = log_pi / torch.log(num_actions)
		entropy = -torch.mean(ent[legal_actions])      # normalized entropy estimate, for logging purposes
	else:
		loss_h  = log_pi.detach() * log_pi
		entropy = -torch.mean(log_pi)

	loss_pi  = torch.mean(loss_pi)
	loss_v   = alpha_v * torch.mean(loss_v)
	loss_h   = alpha_h * torch.mean(loss_h)

	loss = loss_pi + loss_v + loss_h

	# print(f"{loss.item()=} {loss_pi=} {loss_v=} {loss_h=}")
	return loss, loss_pi, loss_v, loss_h, entropy

# this expects concatenated batch of states
def _replay_batch(net, s, raw_a):
	_, v, pi, _ = net(s,  force_action=raw_a)

	v = v.flatten()
	pi = pi.flatten()

	return v, pi

# this version expects the states in sequence
def _replay_lstm(net, s, raw_a, done, hidden_s0):
	ppo_t = len(s)

	v = []
	pi = []

	net.clone_state(hidden_s0)

	for t in range(ppo_t):	
		_, v_, pi_, _ = net(s[t], force_action=raw_a[t])
		net.reset_state(done[t])

		v.append(v_)
		pi.append(pi_)

	v = torch.cat(v).flatten()
	pi = torch.cat(pi).flatten()

	return v, pi

def ppo(s, raw_a, a_cnt, done, v_target, net, gamma, alpha_v, alpha_h, ppo_k, ppo_eps, use_a_t, v_range=None, lstm=False, hidden_s0=None):
	if lstm:
		_, pi_old = _replay_lstm(net, s, raw_a, done, hidden_s0)
	else:
		_, pi_old = _replay_batch(net, s, raw_a)

	pi_old = pi_old.detach()

	# optionally clamp v_target
	v_target_clamped = v_target if v_range is None else v_target.clamp(*v_range)

	loss_avg = []
	entropy_avg = []
	norm_avg = []
	pi_deviations = []

	for i in range(ppo_k):
		if lstm:
			v, pi = _replay_lstm(net, s, raw_a, done, hidden_s0)
		else:
			v, pi = _replay_batch(net, s, raw_a)

		if use_a_t:
			v_adv = torch.clamp(v, 0., None)
			loss_v = (v_target_clamped - v)[a_cnt] ** 2 # only include the situations, where q_continue was used, i.e., skip terminating actions
		else:
			v_adv = v
			loss_v = (v_target_clamped - v) ** 2 

		# loss_pi
		adv = (v_target - v_adv).detach()
		rho = pi / pi_old

		# loss_pi = -adv.detach() * (pi / pi_old)
		# Note: pi is correctly trained only when a_cnt == 1. It is already handled in corresponding net class with "tot_prob[terminate] = .5" line.
		loss_pi = - torch.minimum( rho * adv, torch.clamp(rho, 1-ppo_eps, 1+ppo_eps) * adv )
		pi_deviations.append( rho.mean() )

		# loss_h
		log_pi = torch.log(pi + 1e-10)
		loss_h  = log_pi.detach() * log_pi
		entropy = -torch.mean(log_pi)

		# average & sum
		loss_pi  = torch.mean(loss_pi)
		loss_v   = alpha_v * torch.mean(loss_v)
		loss_h   = alpha_h * torch.mean(loss_h)

		loss = loss_pi + loss_v + loss_h
		
		# perform opt step
		norm = net._update(loss)

		# update stats
		loss_avg.append(loss.item())
		entropy_avg.append(entropy.item())
		norm_avg.append(norm.item())

	return np.mean(loss_avg), np.mean(entropy_avg), np.mean(norm_avg), [x.item() for x in pi_deviations]

	# print(f"{loss.item()=} {loss_pi=} {loss_v=} {loss_h=}")
	# return loss, loss_pi, loss_v, loss_h, entropy
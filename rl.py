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


def ppo(s, raw_a, a_q, v_target, q_target, net, gamma, alpha_v, alpha_q, alpha_h, ppo_k, ppo_eps, v_range=None, num_actions=None):
	# make sure all dimensions are correct
	# print(s.shape[0], raw_a[0].shape , raw_a[1].shape , r.shape)
	_, _, _, pi_old, _ = net(s, force_action=raw_a)
	pi_old = pi_old.flatten().detach()

	# optionally clamp v_target
	v_target_clamped = v_target if v_range is None else v_target.clamp(*v_range)

	loss_avg = []
	entropy_avg = []
	norm_avg = []
	pi_deviations = []

	for i in range(ppo_k):
		_, v, q, pi, _ = net(s, force_action=raw_a)

		v = v.flatten()
		q = q.flatten()
		pi = pi.flatten()

		# loss_v + loss_q
		loss_v = (v_target_clamped - v) ** 2
		loss_q = (q_target - q)[a_q] ** 2 # only include the situations, where q_continue was used, i.e., skip terminating actions

		# loss_pi
		adv = (v_target - v).detach()
		rho = pi / pi_old

		# loss_pi = -adv.detach() * (pi / pi_old)
		loss_pi = - torch.minimum( rho * adv, torch.clamp(rho, 1-ppo_eps, 1+ppo_eps) * adv )
		pi_deviations.append( rho.mean() )

		# loss_h
		log_pi = torch.log(pi + 1e-10)

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

		# average & sum
		loss_pi  = torch.mean(loss_pi)
		loss_v   = alpha_v * torch.mean(loss_v)
		loss_q   = alpha_q * torch.mean(loss_q)
		loss_h   = alpha_h * torch.mean(loss_h)

		loss = loss_pi + loss_v + loss_q + loss_h
		
		# perform opt step
		norm = net._update(loss)

		# update stats
		loss_avg.append(loss.item())
		entropy_avg.append(entropy.item())
		norm_avg.append(norm.item())

	return np.mean(loss_avg), np.mean(entropy_avg), np.mean(norm_avg), [x.item() for x in pi_deviations]

	# print(f"{loss.item()=} {loss_pi=} {loss_v=} {loss_h=}")
	# return loss, loss_pi, loss_v, loss_h, entropy
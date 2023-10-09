import numpy as np
from inverse_dynamics import InvNetwork
from transition_function import reshapeaction_transition
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.autograd import grad
import scipy


alpha = 1.0
beta = 1.0

num_iterations = 500
step_lenth = 0.0001
batch_size = 64

net = InvNetwork()
net.load_state_dict(torch.load('my_network_1.pth'))

for iteration in range(num_iterations):
    s = np.zeros(batch_size, dtype=np.float32)
    # 动作e从正态分布里采，采完之后过tanh限制在0-1之间
    random_sample = np.random.normal(size=batch_size)
    e = (np.tanh(random_sample) + 1) / 2
    s_prime = reshapeaction_transition(e, alpha, beta)  # 实时生成数据
    a = scipy.special.betaincinv(alpha, beta, e)

    alpha_ndarry = alpha * np.ones(batch_size, dtype=np.float32)
    beta_ndarry = beta * np.ones(batch_size, dtype=np.float32)

    s_tensor = torch.tensor(s, requires_grad=True, dtype=torch.float32)
    sp_tensor = torch.tensor(s_prime, requires_grad=True, dtype=torch.float32)
    alpha_tensor = torch.tensor(alpha_ndarry, requires_grad=True, dtype=torch.float32)
    beta_tensor = torch.tensor(beta_ndarry, requires_grad=True, dtype=torch.float32)

    # input_data = torch.tensor([s, s_prime, alpha, beta], dtype=torch.float32)
    e_tensor = torch.tensor(e, dtype=torch.float32).unsqueeze(1)

    mu, sigma = net(torch.stack((s_tensor, sp_tensor, alpha_tensor, beta_tensor),dim=1))

    # 计算 log p(e|s,s',alpha,beta)
    log_likelihood = -0.5 * ((e_tensor - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * torch.log(2 * torch.tensor(np.pi))
    log_likelihood_sum = log_likelihood.sum()

    # 计算 \partial log p(e|s,s',alpha,beta) / \partial alpha
    grad_alpha = grad(log_likelihood_sum, alpha_tensor, retain_graph=True)[0]
    grad_beta = grad(log_likelihood_sum, beta_tensor, retain_graph=True)[0]


    # 4. 计算 J(theta) 的梯度
    alpha_constant = np.log(a) + scipy.special.digamma(alpha+beta) - scipy.special.digamma(alpha)
    beta_costant = np.log(1-a) + scipy.special.digamma(alpha+beta) - scipy.special.digamma(beta)

    logp = log_likelihood.squeeze().detach().numpy()

    di_logp_alpha = grad_alpha.detach().numpy()
    di_logp_beta = grad_beta.detach().numpy()

    J_gradient_alpha = alpha_constant * logp + di_logp_alpha
    J_gradient_beta = beta_costant * logp + di_logp_beta

    J_target = - logp.mean()

    print(f"Iter {iteration} alpha: {alpha:.2f} beta: {beta:.2f} Target: {J_target:.2f}")
    # 5. 更新 theta
    alpha = alpha + step_lenth * J_gradient_alpha.mean()
    beta = beta + step_lenth * J_gradient_beta.mean()



    # 6. 重复收集批次数据并迭代

# 在上述代码中，您需要自行实现与环境的交互、概率计算以及目标函数 J(theta) 的梯度计算函数。
# 请根据具体问题的需求来填充这些函数的实现。

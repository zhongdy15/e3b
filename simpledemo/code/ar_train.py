import numpy as np
from inverse_dynamics import InvNetwork
from transition_function import reshapeaction_transition
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.autograd import grad
import scipy


def policy_evaluation(alpha,beta,batch_size,net):

    # 初始状态为0
    s = np.zeros(batch_size, dtype=np.float32)
    # 内层策略固定为随机均匀分布
    e = np.random.uniform(0, 1, batch_size)
    # 按照当前alpha,beta执行策略并获取下一个状态
    s_prime = reshapeaction_transition(e, alpha, beta)
    # 当前执行的外层动作a = f(e),其中f是参数为alpha,beta的beta incomplete inverse function
    a = scipy.special.betaincinv(alpha, beta, e)

    # alpha值
    alpha_ndarry = alpha * np.ones(batch_size, dtype=np.float32)
    # beta值
    beta_ndarry = beta * np.ones(batch_size, dtype=np.float32)

    s_tensor = torch.tensor(s, requires_grad=True, dtype=torch.float32)
    sp_tensor = torch.tensor(s_prime, requires_grad=True, dtype=torch.float32)
    alpha_tensor = torch.tensor(alpha_ndarry, requires_grad=True, dtype=torch.float32)
    beta_tensor = torch.tensor(beta_ndarry, requires_grad=True, dtype=torch.float32)
    e_tensor = torch.tensor(e, dtype=torch.float32).unsqueeze(1)

    # 逆动力学网络net中,输入[s,s',alpha,beta],输出mu和sigma
    mu, sigma = net(torch.stack((s_tensor, sp_tensor, alpha_tensor, beta_tensor), dim=1))

    # 按照正态概率密度计算 log p(e|s,s',alpha,beta)
    log_likelihood = -0.5 * ((e_tensor - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * torch.log(2 * torch.tensor(np.pi))

    # 目标函数J_target
    logp = log_likelihood.squeeze().detach().numpy()
    J_target = - logp.mean()

    # 计算 \partial log p(e|s,s',alpha,beta) / \partial alpha
    log_likelihood_sum = log_likelihood.sum()
    grad_alpha = grad(log_likelihood_sum, alpha_tensor, retain_graph=True)[0]
    grad_beta = grad(log_likelihood_sum, beta_tensor, retain_graph=True)[0]
    di_logp_alpha = grad_alpha.detach().numpy()
    di_logp_beta = grad_beta.detach().numpy()

    # 计算 beta函数的梯度
    alpha_constant = np.log(a) + scipy.special.digamma(alpha + beta) - scipy.special.digamma(alpha)
    beta_costant = np.log(1 - a) + scipy.special.digamma(alpha + beta) - scipy.special.digamma(beta)


    # 梯度方向
    J_gradient_alpha = -(alpha_constant * logp + di_logp_alpha)
    J_gradient_beta = -(beta_costant * logp + di_logp_beta)

    return J_gradient_alpha.mean(), J_gradient_beta.mean(), J_target



if __name__ == '__main__':
    alpha_initial = 1.
    beta_initial = .5

    alpha = alpha_initial
    beta = beta_initial

    num_iterations = 100
    step_lenth = 0.1
    batch_size = 1024

    net = InvNetwork()
    net.load_state_dict(torch.load('my_network_4.pth'))
    for iteration in range(num_iterations):
        J_gradient_alpha, J_gradient_beta, J_target = policy_evaluation(alpha,beta,batch_size,net)

        print(f"Iter {iteration} alpha: {alpha:.2f} beta: {beta:.2f} gradient_alpha: {J_gradient_alpha.mean():.2f} gradient_beta: {J_gradient_beta.mean():.2f} Target: {J_target:.2f}")
        # 负梯度方向 更新 theta
        # alpha = alpha + step_lenth * J_gradient_alpha.mean()
        # beta = beta + step_lenth * J_gradient_beta.mean()

        if J_gradient_alpha > 0:
            alpha += step_lenth
        else:
            alpha -= step_lenth

        # if J_gradient_beta.mean() > 0:
        #     beta += step_lenth
        # else:
        #     beta -= step_lenth


        if alpha < 0.1:
            alpha = 0.1
        if alpha > 10:
            alpha = 10

        if beta < 0.1:
            beta = 0.1
        if beta > 10:
            beta = 10


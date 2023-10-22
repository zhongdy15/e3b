import numpy as np
from inverse_dynamics import RealInvNetwork
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
    # 当前执行的外层动作a = f(e),其中f是参数为alpha,beta的beta incomplete function
    a = scipy.special.betainc(alpha, beta, e)

    s_tensor = torch.tensor(s, dtype=torch.float32)
    sp_tensor = torch.tensor(s_prime, dtype=torch.float32)

    a_tensor = torch.tensor(a, dtype=torch.float32).unsqueeze(1)

    # 逆动力学网络net中,输入[s,s'],输出mu和sigma
    mu, sigma = net(torch.stack((s_tensor, sp_tensor), dim=1))

    # 按照正态概率密度计算 log p(a|s,s')
    log_likelihood = -0.5 * ((a_tensor - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * torch.log(2 * torch.tensor(np.pi))
    # \log |\frac{\partial f(\theta, e)}{\partial e}|
    #  & =  \log \left(\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha) \Gamma(\beta)} e^{\alpha-1}(1-e)^{\beta-1}\right) \\
    # & =\log \Gamma(\alpha+\beta)-\log \Gamma(\alpha)-\log \Gamma(\beta)+(\alpha-1) \log e +(\beta-1) \log (1 - e)
    epsilon = 1e-8
    beta_log = scipy.special.gammaln(alpha + beta) - scipy.special.gammaln(alpha) - scipy.special.gammaln(beta) \
               + (alpha - 1) * np.log(e + epsilon) + (beta - 1) * np.log(1 - e + epsilon)

    # 目标函数J_target
    logp = log_likelihood.squeeze().detach().numpy()
    J_target = - logp.mean() - beta_log.mean()

    # 计算 beta函数的梯度
    alpha_constant = np.log(e+epsilon) + scipy.special.digamma(alpha + beta) - scipy.special.digamma(alpha)
    beta_costant = np.log(1 - e + epsilon) + scipy.special.digamma(alpha + beta) - scipy.special.digamma(beta)

    #  \left[ 1 - \log P(a |s,s^{\prime}) - \log |\frac{\partial f(\theta,e)}{\partial e}| \right]
    second_term = 1 -logp - beta_log

    # 梯度方向
    J_gradient_alpha = - alpha_constant * second_term
    J_gradient_beta = - beta_costant * second_term

    return J_gradient_alpha.mean(), J_gradient_beta.mean(), J_target



if __name__ == '__main__':
    alpha_initial = 1.
    beta_initial = .5

    alpha = alpha_initial
    beta = beta_initial

    num_iterations = 10000
    step_lenth = 0.1
    batch_size = 1024

    net = RealInvNetwork()
    net.load_state_dict(torch.load('RealInvNetwork.pth'))
    for iteration in range(num_iterations):
        J_gradient_alpha, J_gradient_beta, J_target = policy_evaluation(alpha,beta,batch_size,net)

        print(f"Iter {iteration} alpha: {alpha:.2f} beta: {beta:.2f} gradient_alpha: {J_gradient_alpha.mean():.2f} gradient_beta: {J_gradient_beta.mean():.2f} Target: {J_target:.2f}")
        # 负梯度方向 更新 theta
        # alpha = alpha + step_lenth * J_gradient_alpha.mean()
        # beta = beta + step_lenth * J_gradient_beta.mean()

        if J_gradient_alpha < 0:
            alpha += step_lenth
        else:
            alpha -= step_lenth

        if J_gradient_beta < 0:
            beta += step_lenth
        else:
            beta -= step_lenth


        if alpha < 0.01:
            alpha = 0.01


        if beta < 0.01:
            beta = 0.01


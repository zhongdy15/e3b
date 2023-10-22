import numpy as np
from inverse_dynamics import InvNetwork
from transition_function import reshapeaction_transition
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.autograd import grad
import scipy
import matplotlib.pyplot as plt
from ar_train import policy_evaluation

alpha = .5
beta = .5

net = InvNetwork()
net.load_state_dict(torch.load('my_network_4.pth'))

# 创建一个从0到1之间的一系列x值
batch_size = 100
x = np.linspace(0, 1, batch_size)

transition_value = [reshapeaction_transition(xi,alpha,beta) for xi in x]
transition_value_2 = [reshapeaction_transition(xi) for xi in x]
betaincinv_values = [scipy.special.betaincinv(alpha, beta, xi) for xi in x]

input_data = np.zeros((batch_size, 4), dtype=np.float32)
input_data[:, 0] = 0
input_data[:, 1] = transition_value
input_data[:, 2] = alpha
input_data[:, 3] = beta

input_data = torch.from_numpy(input_data)

mu = net(input_data)[0].squeeze()
sigma = net(input_data)[1].squeeze()
e_tensor = torch.tensor(x)
log_likelihood = -0.5 * ((e_tensor - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * torch.log(2 * torch.tensor(np.pi))
J = - log_likelihood.mean().item()

_,_,J2 = policy_evaluation(alpha,beta,batch_size,net)

predict_mean = mu.tolist()
predict_var = sigma.tolist()
# 创建一个图形窗口
plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.plot(x, predict_mean, label='predict_mean')
plt.plot(x, x, label='real_mean')
plt.xlabel('a')
plt.ylabel('predict_mean')
plt.title('predict_mean a:' + str(alpha)+" b:" +str(beta))
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)

plt.plot(x, predict_var, label='v')
plt.xlabel('a')
plt.ylabel('predict_var')
plt.title('predict_var')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(x, betaincinv_values, label = "f function")
plt.plot(x, transition_value_2, label='trans_a')
plt.plot(x, transition_value, label='trans_e')
plt.xlabel('a/e')
plt.ylabel('s_prime')
plt.title('trans')
plt.grid(True)
plt.legend()

# 显示图形
plt.title("J_target:  {:.2f} J_target_2:  {:.2f}".format(J,J2))
plt.tight_layout()
plt.show()


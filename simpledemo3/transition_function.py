import numpy as np
import matplotlib.pyplot as plt
import scipy

def reshapeaction(e,k1,k2,p):
    assert e >= 0
    assert e <= 1
    assert k1 >= 0
    assert k2 >= 0
    assert p < 1
    assert p >= 0

    if k2 * p - k2 + 1 < k1 * p:
        k2 = (1 - k1 * p) / (1 - p)

    a = 0
    J_a = 0

    if e < p:
        a = k1 * e
        J_a = k1
    elif e > p:
        a = k2 * e - k2 +1
        J_a = k2
    else:
        a = (k1 * e + k2 * e - k2 +1) / 2
        J_a = (k1 + k2 )/2

    return a, J_a

def inversereshapeaction(a,k1,k2,p):

    assert k1 >= 0
    assert k2 >= 0
    assert p < 1
    assert p >= 0

    if k2 * p - k2 + 1 < k1 * p:
        k2 = (1 - k1 * p) / (1 - p)

    e = 0
    J_e = 0

    if a < k1 * p:
        e = 1/k1 * a
        J_e = 1/k1
    elif a > k2 * p - k2 +1:
        e = 1/k2 * a - 1/k2 + 1
        J_e = 1/k2
    else:
        e = p
        J_e = 0

    return e, J_e


def reshapeaction_transition(e,k1=1,k2=1,p=0.5,randomness=0.0):
    a, _ = reshapeaction(e, k1,k2,p)
    k = 2
    acceleration = (a < (0.5/k)) * k * a + (a > (1 - 0.5/k)) *(k*a +1-k) + (a>=(0.5/k)) * (a <= (1 - 0.5/k)) * 0.5
    acceleration += randomness * np.random.normal(size=a.size)
    return acceleration

# def reshapeaction_transition(e,alpha=1.0,beta=1.0,randomness=0.01):
#     a = scipy.special.betainc(alpha, beta, e)
#     acceleration = (a <0.5) * 0 + (a >= 0.5) *((a - 0.5) / 0.5)
#     acceleration += randomness * np.random.normal(size=a.size)
#     return acceleration

# def reshapeaction_transition(e,alpha=1.0,beta=1.0,randomness=0.01):
#     a = scipy.special.betainc(alpha, beta, e)
#     acceleration = (a <0.5) * 2*a + (a >= 0.5) * 1
#     acceleration += randomness * np.random.normal(size=a.size)
#     return acceleration

if __name__ == '__main__':
    # 定义分布的形状参数
    k1=0.5
    k2=0.5
    p=0.2

    # 创建一个从0到1之间的一系列x值
    x = np.linspace(0, 1, 100)

    # 计算不完全Beta函数（betainc）的值 作为f函数
    betainc_values = [reshapeaction(xi,k1,k2,p)[0] for xi in x]

    # 计算不完全Beta函数的逆（betaincinv）的值
    betaincinv_values = [inversereshapeaction(xi,k1,k2,p)[0] for xi in x]

    # 创建一个图形窗口
    plt.figure(figsize=(15, 5))

    # 绘制不完全Beta函数（betainc）的图像
    plt.subplot(1, 3, 1)
    plt.plot(x, betainc_values, label='a = f(e)')
    plt.xlabel('e')
    plt.ylabel('f function')
    plt.title('f function')
    plt.grid(True)
    plt.legend()

    # 绘制不完全Beta函数的逆（betaincinv）的图像
    plt.subplot(1, 3, 2)
    plt.plot(x, betaincinv_values, label='e = h(a)')
    plt.xlabel('a')
    plt.ylabel('h function')
    plt.title('h function')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    transition_value = [reshapeaction_transition(xi) for xi in x]
    plt.plot(x, transition_value, label='original_trans_a')
    transition_value_2 = [reshapeaction_transition(xi,k1,k2,p) for xi in x]
    plt.plot(x, transition_value_2, label=f'trans_with k1{k1:.2f} k2={k2:.2f} p={p:.2f}')
    plt.xlabel('a/e')
    plt.ylabel('s_prime')
    plt.title('trans')
    plt.grid(True)
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()


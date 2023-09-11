import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from log_datas import read_data_from_logger
# sns.set_style('whitegrid')
sns.set_style("ticks")
import matplotlib
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

def smoother(x, a=0.9, w=10, mode="moving"):
    if mode == "moving":
        y = [x[0]]
        for i in range(1, len(x)):
            y.append((1 - a) * x[i] + a * y[i - 1])
    elif mode == "window":
        y = []
        for i in range(len(x)):
            y.append(np.mean(x[max(i - w, 0):i + 1]))
    else:
        raise NotImplementedError
    return y


def plot_title(title=None):
    plt.figure(figsize=(6.6, 4.4), dpi=300)
    plt.xlabel("Million Steps", {'size': 23})
    plt.ylabel("Episode Return", {'size': 23})
    plt.title(title, {'size': 23})
    plt.tick_params(labelsize=23)

def plot_curve(index, dfs, label=None, shaded_err=False, shaded_std=True, insert=None):
    color = sns.color_palette()[index]
    N = np.min([len(df["exploration/num steps total"]) for df in dfs])
    x = dfs[0]["exploration/num steps total"][:N] / 1e6
    ys = [smoother(df["evaluation/Average Returns"][:N], w=60, mode="window") for df in dfs] # w越大，曲线越平滑
    if insert is not None:
        x = np.insert(x,0,insert[0])
        for i in range(len(ys)):
            ys[i] = np.insert(ys[i],0,insert[1])


    y_mean = np.mean(ys, axis=0)
    y_std = np.std(ys, axis=0) /1
    y_stderr = y_std / np.sqrt(len(ys))
    if label == 'HORL':
        label = 'IQL+LPD'
    if label is None:
        lin = plt.plot(x, y_mean, color=color)
    else:
        lin = plt.plot(x, y_mean, color=color, label=label)
    if shaded_err:
        plt.fill_between(x, y_mean - y_stderr, y_mean + y_stderr, color=color, alpha=.2)
    if shaded_std:
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=.2)
    # leg = plt.legend(loc=1, prop={'size': 21}, frameon=False, bbox_to_anchor=(1.65, 1.10), ncol=1)
    leg = plt.legend(loc=4, prop={'size': 21}, frameon=True,ncol=1)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.5)
    return lin

# 指定CSV文件的路径
path = "D:\\research\\e3b_remotelog\\0911算法结果对比\\"
algo_csv_list = [["elliptical\\env_MiniHack-MultiRoom-N6-v0-eb_elliptical-icm-lr_0.0001-plr_0.0001-entropy_0.005-intweight_1.0-ridge_0.1-cr_1-rn_int-seed_1"],
                 ["impala\\env_MiniHack-MultiRoom-N6-v0model_vanilla-lr_0.0001-entropy_0.005-seed_1"],
                 ["icm\\env_MiniHack-MultiRoom-N6-v0model_curiosity-lr_0.0001-fc_1.0-ic_0.1-entropy_0.005-intweight_0.1-seed_1"],
                 ["rnd\\env_MiniHack-MultiRoom-N6-v0model_rnd-lr_0.0001-entropy_0.005-intweight_0.001-seed_1"],
                 ]
# algo_name = algo_name_list[3]
# csv_file_path = os.path.join(path,algo_name,"logs.csv")

# 用于存储提取的数据的列表, algo_data_list[0]是一个list，包含了所有该algo生成的不同种子的文件
algo_data_list = [ [] for _ in range(len(algo_csv_list))]



# 第i个算法
for i in range(len(algo_csv_list)):
    # 第j个种子
    for j in range(len(algo_csv_list[i])):
        frames_list = []
        mean_episode_return_list = []
        csv_name = algo_csv_list[i][j]
        csv_file_path =os.path.join(path,csv_name,"logs.csv")
        with open(csv_file_path, mode='r') as file:
            # 使用csv.DictReader来读取CSV文件，并指定字段名称
            reader = csv.DictReader(file)

            # 遍历CSV文件的每一行
            for row in reader:
                # 提取frames字段和mean_episode_return字段的值
                frames = int(row['frames'])
                mean_episode_return = float(row['mean_episode_return'])

                # 将提取的值添加到相应的列表中
                if not np.isnan(mean_episode_return):
                    frames_list.append(frames)
                    mean_episode_return_list.append(mean_episode_return)
        frames_array = np.array(frames_list)
        mean_episode_return_array = np.array(mean_episode_return_list)
        frames_return_dict={"exploration/num steps total": frames_array,
                            "evaluation/Average Returns": mean_episode_return_array}
        algo_data_list[i].append(frames_return_dict)

experiment_name = algo_csv_list[i][0].split("env_")[1].split("-v0")[0]
plot_title(experiment_name)
color = 0
for i in range(len(algo_csv_list)):
    name = algo_csv_list[i][0].split("\\")[0]
    plot_curve(color, algo_data_list[i],name )
    color += 1
plt.savefig(experiment_name+".pdf", dpi=300, bbox_inches='tight')


import csv
import matplotlib.pyplot as plt
import os

# 指定CSV文件的路径
path = "D:\\research\\e3b_remotelog\\0911算法结果对比\\elliptical\\"
algo_name = "env_MiniHack-MultiRoom-N6-v0-eb_elliptical-icm-lr_0.0001-plr_0.0001-entropy_0.005-intweight_1.0-ridge_0.1-cr_1-rn_int-seed_1"
csv_file_path = os.path.join(path,algo_name,"logs.csv")

# 用于存储提取的数据的列表
frames_list = []
mean_episode_return_list = []

try:
    with open(csv_file_path, mode='r') as file:
        # 使用csv.DictReader来读取CSV文件，并指定字段名称
        reader = csv.DictReader(file)

        # 遍历CSV文件的每一行
        for row in reader:
            # 提取frames字段和mean_episode_return字段的值
            frames = int(row['frames'])
            mean_episode_return = float(row['mean_episode_return'])

            # 将提取的值添加到相应的列表中
            frames_list.append(frames)
            mean_episode_return_list.append(mean_episode_return)

    # 绘制图像
    plt.figure(figsize=(10, 6))
    plt.plot(frames_list, mean_episode_return_list, marker='o', linestyle='-')
    plt.title('Mean Episode Return vs. Frames')
    plt.xlabel('Frames')
    plt.ylabel('Mean Episode Return')
    plt.grid(True)
    plt.show()

except FileNotFoundError:
    print(f"File not found: {csv_file_path}")
except Exception as e:
    print(f"An error occurred: {str(e)}")

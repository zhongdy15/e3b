import logging
import os
import numpy as np
import time


def make_dirs(algo_name, experiment_name):
    file_path = f"z_data/{algo_name}-{experiment_name}"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path


def make_logger(algo_name, experiment_name, experiment_start_time):
    file_path = make_dirs(algo_name, experiment_name)
    file_name = f'{file_path}/{experiment_start_time}_logger.txt'
    logger = DataLog(file_name)
    return logger, file_path, file_name


class DataLog(object):
    def __init__(self, file_log_name):
        test = TestLog(file_log_name)
        self.file_log_name = file_log_name
        self.logger = test.get_logger()

    def add_data(self, cur_step, reward):
        self.logger.info(f"exploration/num steps total: |{cur_step: 8d}|, evaluation/Average Returns: |{reward: 8.3f}|")


class TestLog(object):
    # 灏佽logging
    def __init__(self, file_log_name, logger=None):
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)
        self.log_time = time.strftime("%Y_%m_%d_")
        self.log_name = file_log_name

    def set_logger(self):
        if not self.logger.handlers:                           # 鍒ゆ柇濡傛灉handlers涓棤handler鍒欐坊鍔犳柊鐨刪andler
            self.fh = logging.FileHandler(self.log_name, "a")  # 鍒涘缓灏嗘棩蹇楀啓鍏ュ埌鏂囦欢锛宎琛ㄧず浠ヨ拷鍔犵殑褰㈠紡鍐欏叆鏃ュ織
            self.fh.setLevel(logging.DEBUG)
            self.chd = logging.StreamHandler()                 # 鍒涘缓浠庢帶鍒跺彴杈撳叆鏃ュ織
            self.chd.setLevel(logging.DEBUG)     # 璁剧疆涓簄otset锛屽彲浠ユ墦鍗癲ebug銆乮nfo銆亀arning銆乪rror銆乧ritical鐨勬棩蹇楃骇鍒?
            self.formatter = logging.Formatter("%(asctime)s | %(message)s")
            self.fh.setFormatter(self.formatter)
            self.chd.setFormatter(self.formatter)
            self.logger.addHandler(self.fh)                    # 娣诲姞鏂囦欢鏃ュ織鐨勬棩蹇楀鐞嗗櫒
            self.logger.addHandler(self.chd)                   # 娣诲姞鎺у埗鍙扮殑鏃ュ織澶勭悊鍣?

    def get_logger(self):
        TestLog.set_logger(self)
        return self.logger

    def remove_log_handler(self):
        # 绉婚櫎handlers涓殑鍏冪礌
        self.logger.removeHandler(self.fh)             # 绉婚櫎鍙ユ焺锛岄伩鍏嶉噸澶嶆墦鍗扮浉鍚岀殑鏃ュ織
        self.logger.removeHandler(self.chd)            # 绉婚櫎鍙ユ焺锛岄伩鍏嶉噸澶嶆墦鍗扮浉鍚岀殑鏃ュ織
        self.fh.close()                                # 鍏抽棴鏃ュ織澶勭悊鍣?
        self.chd.close()                               # 鍏抽棴鏃ュ織澶勭悊鍣?


def read_data_from_logger(file_log_name):
    steps = []
    rewards = []
    with open(file_log_name, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 鏁磋璇诲彇鏁版嵁
            if not lines:
                break
                pass
            split_data = lines.split('|')
            steps.append(float(split_data[2].lstrip()))
            rewards.append(float(split_data[4]))
            pass
    df = {'exploration/num steps total': np.asarray(steps), 'evaluation/Average Returns': np.asarray(rewards)}
    return df





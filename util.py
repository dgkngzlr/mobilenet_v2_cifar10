from datetime import datetime
import os
import glob


class Logger:
    epoch_index = 0
    step_index = 0
    train_info_index = 0
    lr_index = 0

    @classmethod
    def save_epoch_loss(cls, loss):
        with open("./log/train_epoch_loss.csv", "a", encoding="utf-8") as f:
            f.write(f"{cls._get_time()},{cls.epoch_index},{loss}\n")

        cls.epoch_index += 1

    @classmethod
    def save_step_loss(cls, loss):
        with open("./log/train_step_loss.csv", "a", encoding="utf-8") as f:
            f.write(f"{cls._get_time()},{cls.step_index},{loss}\n")

        cls.step_index += 1

    @classmethod
    def save_train_info(cls, prec, recall, acc):
        with open("./log/train_info.csv", "a", encoding="utf-8") as f:
            f.write(f"{cls._get_time()},{cls.train_info_index},{prec},{recall},{acc}\n")

        cls.train_info_index += 1

    @classmethod
    def save_test_info(cls, prec, recall, acc):
        with open("./log/test_info.csv", "a", encoding="utf-8") as f:
            f.write(f"{cls._get_time()},{cls.train_info_index},{prec},{recall},{acc}\n")

    @classmethod
    def save_lr(cls, lr):
        with open("./log/train_lr.csv", "a", encoding="utf-8") as f:
            f.write(f"{cls._get_time()},{cls.lr_index},{lr}\n")

        cls.lr_index += 1

    @classmethod
    def _get_time(cls):
        now = datetime.now()

        return f"{now.day}:{now.month}:{now.year}:{now.hour}:{now.minute}:{now.second}"

    @classmethod
    def clear_all(cls):
        file_list = glob.glob('./log/*.csv')
        for file in file_list:
            os.remove(file)

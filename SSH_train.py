import os
import cv2
import time
import numpy as np

from mindspore.nn import SGD
from mindspore import context, TimeMonitor, Model, load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint

from SSH_model import SSHModel
from src.model_utils.config import config
from src.network_define import LossNet, WithLossCell, TrainOneStepCell, LossCallBack
from src.dataset import data_to_mindrecord_byte_image, create_ssh_dataset

rank = 0
device_num = 1


def get_dataset():
    print("Start create dataset!")

    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is SSHmindrecord0, 1, ... file_num.
    prefix = "SSH.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    print("CHECKING MINDRECORD FILES ...")

    if rank == 0 and not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        # if config.dataset == "coco":
        #     if os.path.isdir(config.coco_root):
        #         if not os.path.exists(config.coco_root):
        #             print("Please make sure config:coco_root is valid.")
        #             raise ValueError(config.coco_root)
        #         print("Create Mindrecord. It may take some time.")
        #         data_to_mindrecord_byte_image(config, "coco", True, prefix)
        #         print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        #     else:
        #         print("coco_root not exits.")
        # else:
        if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
            if not os.path.exists(config.image_dir):
                print("Please make sure config:image_dir is valid.")
                raise ValueError(config.image_dir)
            print("Create Mindrecord. It may take some time.")
            data_to_mindrecord_byte_image(config, prefix)
            print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        else:
            print("image_dir or anno_path not exits.")

    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    print("CHECKING MINDRECORD FILES DONE!")
    # When create MindDataset, using the fitst mindrecord file, such as SSH.mindrecord0.
    dataset = create_ssh_dataset(config, mindrecord_file, batch_size=config.batch_size,
                                 device_num=device_num, rank_id=rank,
                                 num_parallel_workers=config.num_parallel_workers,
                                 python_multiprocessing=config.python_multiprocessing)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done!")

    return dataset_size, dataset


def train_ssh():
    dataset_size, dataset = get_dataset()

    net = SSHModel(config)
    net = net.set_train()

    param_dict = load_checkpoint('./ckpts/ckpt_0_1/ssh_2-5_50.ckpt')
    keys = [key for key in param_dict]
    load_param_into_net(net, param_dict)

    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(mstype.float16)

    loss = LossNet()
    opt = SGD(params=net.trainable_params(), learning_rate=config.lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    net_with_loss = WithLossCell(net, loss)
    net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)

    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(per_print_times=dataset_size, rank_id=rank)
    ckptconfig = CheckpointConfig(save_checkpoint_steps=dataset_size,
                                  keep_checkpoint_max=4)
    save_checkpoint_path = os.path.join(config.save_checkpoint_path, "ckpt_" + str(rank) + "/")
    ckpoint_cb = ModelCheckpoint(prefix='ssh', directory=save_checkpoint_path, config=ckptconfig)
    cb = [time_cb, loss_cb, ckpoint_cb]

    model = Model(net)
    model.train(config.epoch_size, dataset, callbacks=cb)


if __name__ == '__main__':
    train_ssh()

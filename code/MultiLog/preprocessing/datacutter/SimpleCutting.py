from CONSTANTS import *


def cut_by_613(instances, time_index2block):
    print("start cut_by_613")
    sorted_keys = sorted(time_index2block.keys())
    dev_split = int(0.01 * len(time_index2block))
    train_split = int(0.89 * len(time_index2block))

    train_and_dev_time_index = sorted_keys[:(train_split + dev_split)]
    train_time_index = train_and_dev_time_index[:]
    dev_time_index = train_and_dev_time_index[train_split:]
    test_time_index = sorted_keys[(train_split + dev_split):]

    print("start get train")
    train = get_by_blocks(instances, get_all_blocks(time_index2block, train_time_index))
    print("start shuffle")
    np.random.shuffle(train)
    print("start get dev")
    dev = get_by_blocks(instances, get_all_blocks(time_index2block, dev_time_index))
    print("start get test")
    test = get_by_blocks(instances, get_all_blocks(time_index2block, test_time_index))
    return train, dev, test


def get_all_blocks(time_index2block, time_indexs):
    result = []
    for time_index in time_indexs:
        result.extend(time_index2block[time_index])
    return result


def get_by_blocks(instances, block_list):
    result = []
    for instance in instances:
        if instance.id in block_list:
            result.append(instance)
    return result

import pandas as pd

config = {}
class TrainingConfig(object):
    p = 0.9
    base_rate = 1e-2
    momentum = 0.9
    decay_step = 15000
    decay_rate = 0.95
    epoches = 10
    evaluate_every = 100
    checkpoint_every = 100


class ModelConfig(object):
    conv_layers = [[256, 7, 3],
                   [256, 7, 3],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, 3]]

    fully_connected_layers = [1024, 1024]
    th = 1e-6


def is_chinese(check_str):
    """
    judge if is chinese
    """
    for str_item in check_str:
        if u'\u4e00' <= str_item <= u'\u9fff':
            return True
    return False

def allchars():

    path =  "../Data/data.csv"
    data = pd.read_csv(path, sep="\t", encoding="utf-8")
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    charall = list()
    for i in xrange(len(data["text"])):
        for char in data["text"][i]:
            if is_chinese(char) or alphabet.__contains__(char):
                if char not in charall:
                    charall.append(char)
    return charall


    
class Config(object):

    alphabet = allchars()
    alphabet_size = len(alphabet)
    l0 = 1014
    batch_size = 64
    no_of_classes = 3

    train_data_source = "../Data/data.csv"
    dev_data_source = '../Data/testdata.csv'

    training = TrainingConfig()
    model = ModelConfig()






config = Config()

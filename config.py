import configparser
import os

class Config(object):
    def __init__(self, config_file):

        conf = configparser.ConfigParser()
        # # 尝试读取配置文件 config_file，如果失败则打印加载失败的信息
        try:
            conf.read(config_file)
        except:
            # 失败则打印信息
            print("loading config: %s failed" % (config_file))

        # Hyper-parameter
        self.epochs = conf.getint("Model_Setup", "epochs")
        self.lr = conf.getfloat("Model_Setup", "lr")
        self.weight_decay = conf.getfloat("Model_Setup", "weight_decay")
        self.k = conf.getint("Model_Setup", "k")
        self.dropout = conf.getfloat("Model_Setup", "dropout")
        self.no_cuda = conf.getboolean("Model_Setup", "no_cuda")
        self.no_seed = conf.getboolean("Model_Setup", "no_seed")
        self.seed = conf.getint("Model_Setup", "seed")
        # self.drop_rate = conf.getfloat("Model_Setup", "drop_rate")
        # self.add_rate = conf.getfloat("Model_Setup", "add_rate")
        self.p = conf.getfloat("Model_Setup", "p")
        self.lambd = conf.getfloat("Model_Setup", "lambd")
        

        # Dataset
        # 从配置文件中读取文件路径参数
        self.n = conf.getint("Data_Setting", "n")
        self.fdim = conf.getint("Data_Setting", "fdim")
        self.class_num = conf.getint("Data_Setting", "class_num")
        self.structgraph_path = conf.get("Data_Setting", "structgraph_path")
        self.featuregraph_path = conf.get("Data_Setting", "featuregraph_path")
        self.feature_path = conf.get("Data_Setting", "feature_path")
        self.label_path = conf.get("Data_Setting", "label_path")
        self.test_path = conf.get("Data_Setting", "test_path")
        self.train_path = conf.get("Data_Setting", "train_path")
        self.val_path = conf.get("Data_Setting", "val_path")

import time, os, yaml, torch
from easydict import EasyDict as edict


def get_config(config_file):
    """ Construct and snapshot hyper parameters """
    # config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))
    config = edict(yaml.load(open(config_file, 'r')))

    # create hyper parameters
    config.run_id = str(os.getpid())
    config.exp_name = '_'.join([
        config.model.name, config.dataset.name,
        time.strftime('%Y-%b-%d-%H-%M-%S'), config.run_id
    ])

    save_dir = os.path.join(config.exp_dir, config.exp_name)
    save_name = os.path.join(save_dir, 'config.yaml')
    # snapshot hyperparameters
    mkdir(config.exp_dir)
    mkdir(save_dir)

    yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

    return config


def edict2dict(edict_obj):
    dict_obj = {}

    for key, vals in edict_obj.items():
        if isinstance(vals, edict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals

    return dict_obj


def mkdir(folder):
  if not os.path.isdir(folder):
    os.makedirs(folder)


def set_device(config):
    if torch.cuda.is_available() and config.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
        config.device = 'cuda:0'
        # print('use gpu indexed: %d' % gpu)
    else:
        config.gpu = -1
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        config.device = 'cpu'
        # print('use cpu')

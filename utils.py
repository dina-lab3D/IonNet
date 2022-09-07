import operator as op
from functools import reduce

from wandb import Config

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom


def fix_dict_in_config(wandb):
    """
    fix sweep yaml to override the default config.
    Config uses regular names for example inside the training dictionary there's the epochs key
    however when converting the sweep_conv.yaml to a dictionary it adds those nested dictionaries as dots for example
    trainining_dict.epochs. This new key won't override the old default epochs so in this case we have to parse the
    "dotted" keys and create inner dictionaries for the sweep to actually check different values.
    @param wandb:
    @return:
    """
    config = dict(wandb.config)
    for k, v in config.copy().items():
        if '.' in k:
            new_key = k.split('.')[0]
            inner_key = k.split('.')[1]
            if new_key not in config.keys():
                config[new_key] = {}
            config[new_key].update({inner_key: v})
            del config[k]

    wandb.config = Config()
    for k, v in config.items():
        wandb.config[k] = v
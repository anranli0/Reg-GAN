#!/usr/bin/python3

import argparse
import os
# from trainer import Cyc_Trainer,Nice_Trainer,P2p_Trainer,Munit_Trainer,Unit_Trainer, Reg_Trainer
from trainer import Reg_Trainer
import yaml
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# assert torch.cuda.is_available()
torch.cuda.set_device(0)

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/RegGan.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    
    if config['name'] == 'RegGan':
        trainer = Reg_Trainer(config)
    # if config['name'] == 'CycleGan':
    #     trainer = Cyc_Trainer(config)
    # elif config['name'] == 'RegGan':
    #     trainer = Reg_Trainer(config)
    # elif config['name'] == 'Munit':
    #     trainer = Munit_Trainer(config)
    # elif config['name'] == 'Unit':
    #     trainer = Unit_Trainer(config)
    # elif config['name'] == 'NiceGAN':
    #     trainer = Nice_Trainer(config)
    # elif config['name'] == 'P2p':
    #     trainer = P2p_Trainer(config)
    trainer.train()
     
    



###################################
if __name__ == '__main__':
    main()
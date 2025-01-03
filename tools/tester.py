#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import numpy as np
from tqdm import tqdm

from models.mlnet.mlnet import MLnet


class Tester(object):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)

        if args.backbone == 'mlnet':
            print(f'Using backbone {args.backbone}')
            self.model = MLnet([config['mlnet_input_size'][1], config['mlnet_input_size'][0]])

        elif args.backbone == 'tasednet':
            print(f'Using backbone {args.backbone}')

        elif args.backbone == 'transalnet':
            print(f'Using backbone {args.backbone}')

        else:
            sys.exit('A backbone must be specified!')


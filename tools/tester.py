#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
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
            checkpoint_path = config['mlnet_model_path']
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['model'])

        elif args.backbone == 'tasednet':
            print(f'Using backbone {args.backbone}')

        elif args.backbone == 'transalnet':
            print(f'Using backbone {args.backbone}')

        else:
            sys.exit('A backbone must be specified!')

        self.model.to(self.device)
        self.model.eval()

    def test_mlnet(self, test_dataloader, ):
        print('Testing with MLnet....')
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader)
            for _, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                outputs = self.model(batch['rgb'])
                outputs = outputs.cpu().numpy() if outputs.is_cuda else outputs.detach().numpy()
                outputs = np.squeeze(outputs)

                # for i, output in enumerate(outputs):
                #     output = output / np.max(outputs + 1e-6) * 255
                #     print(f'Saving heatmap {i}...')
                #     cv2.imwrite(output, )


                #TODO: Add the metrics computing here based on 'anno'
                #TODO: Evaluating whether need to resize the output back the input size, need invert padding.


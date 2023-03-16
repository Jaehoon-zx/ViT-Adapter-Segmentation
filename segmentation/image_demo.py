# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp
import glob


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('img_dir', help='Image file')
    parser.add_argument('val_list', help='Validation list')

    parser.add_argument('--out', type=str, default="inference", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)
    
    f = open(args.val_list, 'r')
    for line in f.readlines():
        line = line.strip()
        # for img in glob.glob(args.img_dir + line + '.jpeg'):
        img = args.img_dir + line + '.jpeg'
        # test a single image
        result = inference_segmentor(model, img)
        # show the results
        if hasattr(model, 'module'):
            model = model.module
        # img = model.show_result(args.img, result,
        #                         palette=get_palette(args.palette),
        #                         show=False, opacity=args.opacity)

        classes = ('cloudy', 'uncertain clear', 'probably clear', 'confident clear')
        palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 0]]
        # RED, GREEN, BLUE, YELLOW
        
        result = model.show_result(img, result,
                                palette = palette,
                                show=False, opacity=args.opacity)

        mmcv.mkdir_or_exist(args.out)
        out_path = osp.join(args.out, osp.basename(img))
        cv2.imwrite(out_path, result)
        print(f"Result is save at {out_path}")
        # changed

if __name__ == '__main__':
    main()
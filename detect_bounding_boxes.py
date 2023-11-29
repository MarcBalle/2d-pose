import argparse
import os

import cv2
from tqdm import tqdm
import torch

from mmdet.apis import DetInferencer

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', '-v', type=str, required=True, help='Path to the input video')
    parser.add_argument('--frames', '-f', type=str, default='', help='Number of frames to analyze')
    parser.add_argument('--config', '-c', type=str, required=True, help='mmdetection model config file')
    parser.add_argument('--checkpoint', '-ckpt', type=str, required=True, help='pth checkpoint file')
    parser.add_argument('--output', '-o', type=str, default='.', help='Output directory')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.isdir(os.path.join(args.output, 'output')):
        os.mkdir(os.path.join(args.output, 'output'))

    # Initialize the DetInferencer
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    inferencer = DetInferencer(model=args.config, device=device)
    
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_length = int(args.frames) if args.frames else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

     # Read first frame and perform inference
    _, img = cap.read()
    results = inferencer(img, return_vis=True)
    bb_img = results['visualization'][0]
    
    h, w = img.shape[:2]
    size = (w, h)
    out = cv2.VideoWriter(os.path.join(args.output, 'bb_output.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    out.write(bb_img)

    for i in tqdm(range(1, video_length)):
        # Image read in BGR order (required by mmdetections)
        _, img = cap.read()

        # Perform inference
        results = inferencer(img, return_vis=True)
        bb_img = results['visualization'][0]
        out.write(bb_img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
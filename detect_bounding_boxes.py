import cv2
from tqdm import tqdm
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', '-v', type=str, required=True, help='path to the input video')
    parser.add_argument('--frames', '-f', type=str, default='', help='number of frames to analyze')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()
    
    cap = cv2.VideoCapture(args.video)

    video_length =  int(args.frames) if args.frames else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(video_length)):
        # Image read in BGR order (required by mmdetections)
        _, img = cap.read()
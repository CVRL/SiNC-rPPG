import numpy as np
import json
import argparse
import os
import sys

sys.path.append('../')
import utils


def ls(x='.'):
    return sorted(os.listdir(x))

def join(*x):
    return os.path.join(*x)

def read_ground_truth(json_path):
    with open(json_path, 'r') as infile:
        gt_data = json.load(infile)
    ## Read video timestamps
    video_t = []
    for sample in gt_data['/Image']:
        video_t.append(sample['Timestamp'])
    ## Read oximeter data
    wave_t = []
    wave = []
    for sample in gt_data['/FullPackage']:
        wave_t.append(sample['Timestamp'])
        wave.append(sample['Value']['waveform'])
    video_t = np.array(video_t)*1e-9
    wave_t = np.array(wave_t)*1e-9
    wave = np.array(wave)
    wave = np.interp(video_t, wave_t, wave)
    return video_t, wave


def main(args):

    input_root = args.input
    output_root = args.output
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    sessions = ls(input_root)
    for session in sessions:
        session_dir = join(input_root, session)
        gt_path = join(session_dir, f'{session}.json')
        frame_dir = join(session_dir, session)
        output_path = join(output_root, f'{session}.npz')
        wave_t, wave = read_ground_truth(gt_path)
        num_frames = len(ls(frame_dir))
        print('t,wave,n_frames:', wave_t.shape, wave.shape, num_frames)
        lmrks = utils.mediapipe_landmark_directory(frame_dir)
        output_video, successful = utils.make_video_array_from_directory(frame_dir, lmrks)
        if successful:
            print('video shape: ', output_video.shape)
            print('lmrks shape: ', lmrks.shape)
            print('waves shape: ', wave.shape)
            print(output_path)
            np.savez_compressed(output_path, video=output_video, wave=wave, video_path=frame_dir, fps=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        help='Path to the original PURE dataset directory.')
    parser.add_argument('output',
                        help='Path to the preprocessed output dataset directory with cropped faces.')
    args = parser.parse_args()
    main(args)


import pandas as pd
import argparse
from natsort import natsorted
import os


def main(args):

    input_root = args.input
    output_path = args.output

    samples = natsorted(os.listdir(input_root))
    subj_ids = []
    sess_ids = []
    paths = []
    for sample in samples:
        print(sample)
        # example: 09-03.npz
        subject, session = os.path.splitext(sample)[0].split('-')
        sample_path = os.path.join(input_root, sample)
        sample_path = os.path.abspath(sample_path) #use absolute path
        print(sample, subject, session, sample_path)

        paths.append(sample_path)
        subj_ids.append(subject)
        sess_ids.append(session)

    d = {}
    d['subj_id'] = subj_ids
    d['sess_id'] = sess_ids
    d['path'] = paths

    df = pd.DataFrame(d)
    df.to_csv(output_path, index=False)
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        help='Path to the preprocessed output dataset directory with cropped faces.')
    parser.add_argument('output',
                        help='Path to the metadata csv which has paths to the preprocessed data.')
    args = parser.parse_args()
    main(args)

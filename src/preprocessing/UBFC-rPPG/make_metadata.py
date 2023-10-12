import pandas as pd
import argparse
from natsort import natsorted
import os


def main(args):

    input_root = args.input
    output_path = args.output

    subjects = natsorted(os.listdir(input_root))
    ids = []
    paths = []
    for subject in subjects:
        subject_id = int(subject[7:-4])
        print(subject, subject_id)
        subject_path = os.path.join(input_root, subject)
        subject_path = os.path.abspath(subject_path)
        paths.append(subject_path)
        ids.append(subject_id)

    d = {}
    d['id'] = ids
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

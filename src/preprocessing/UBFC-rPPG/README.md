## Preprocessing the UBFC-rPPG Dataset
1.) To crop all of the videos to face landmarks from MediaPipe and downscale to 64x64 pixels for the 3D-CNNs:

``
python make_dataset.py <path-to-downlaoded-UBFC> <path-to-preprocessed-UBFC>
``

2.) Then make the metadata file which contains paths to all of the preprocessed data:

``
python make_metadata.py <path-to-preprocessed-UBFC> ../../datasets/metadata/UBFC.csv
``

After completing step 2, you can run train.py and test.py with the UBFC dataset.

## Preprocessing the PURE Dataset
1.) To crop all of the videos to face landmarks from MediaPipe and downscale to 64x64 pixels for the 3D-CNNs:

``
python make_dataset.py <path-to-downlaoded-PURE> <path-to-preprocessed-PURE>
``

2.) Then make the metadata file which contains paths to all of the preprocessed data:

``
python make_metadata.py <path-to-preprocessed-PURE> ../../datasets/metadata/PURE.csv
``

After completing step 2, you can run train.py and test.py with the PURE dataset.

import numpy as np
import mediapipe as mp
import cv2
import os

OUT_IMG_HEIGHT = 64
OUT_IMG_WIDTH = 64

## This is a mapping from Mediapipe's FaceMesh coordinates to OpenFace's coordinates
CANONICAL_LMRKS = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]


def face_mesh_to_array(results, img_w, img_h):
    if not results.multi_face_landmarks:
      lmrks = None
    else:
      lmrks = np.array([[results.multi_face_landmarks[0].landmark[i].x,
                         results.multi_face_landmarks[0].landmark[i].y]
                        for i in CANONICAL_LMRKS])
      lmrks = (lmrks * [img_w, img_h]).astype(int)
    return lmrks


def mediapipe_landmark_directory(frame_dir):
    all_lmrks = []
    prev_lmrks = np.zeros((68, 2), dtype=np.int32)
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5)
    frame_files = sorted(os.listdir(frame_dir))
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = frame.shape
        results = face_mesh.process(frame)
        lmrks = face_mesh_to_array(results, img_w, img_h)
        if lmrks is not None:
            prev_lmrks = lmrks
        else:
            lmrks = prev_lmrks
        all_lmrks.append(lmrks)
    all_lmrks = np.stack(all_lmrks)
    return all_lmrks


def mediapipe_landmark_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_lmrks = []
    prev_lmrks = np.zeros((68, 2), dtype=np.int32)
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        ## Calculate landmarks with face_mesh
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = frame.shape
        results = face_mesh.process(frame)
        lmrks = face_mesh_to_array(results, img_w, img_h)
        if lmrks is not None:
            prev_lmrks = lmrks
        else:
            lmrks = prev_lmrks
        all_lmrks.append(lmrks)
    all_lmrks = np.stack(all_lmrks)
    return all_lmrks


def make_video_array_from_directory(vid_dir, lmrks, w=OUT_IMG_WIDTH, h=OUT_IMG_HEIGHT, dtype=np.uint8):
    vid_len = len(lmrks)
    video_idx = 0
    frame_list = [os.path.join(vid_dir, f) for f in sorted(os.listdir(vid_dir))]
    output_video = np.zeros((vid_len, h, w, 3), dtype=dtype)
    successful = True

    for frame_path in frame_list:
        frame = cv2.imread(frame_path)
        if video_idx == 0:
            img_h, img_w = frame.shape[:2]

        if video_idx < vid_len:
            lmrk = lmrks[video_idx]
        else: #lmrks are shorter than video
            successful = False
            print('ERROR: Fewer landmarks than video frames, must relandmark.')
            break

        lmrk = lmrk.astype(int)
        bbox = get_bbox(lmrk, img_w, img_h)
        square_bbox = get_square_bbox(bbox, img_w, img_h)

        x1,y1,x2,y2 = square_bbox
        cropped = frame[y1:y2, x1:x2]
        if cropped.size < 1:
            resized = np.zeros((h, w, 3), dtype=cropped.dtype)
        else:
            resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
        output_video[video_idx] = resized
        video_idx += 1

    if video_idx < vid_len:
        successful = False
        print(f'ERROR: Reached video idx {video_idx} while video was expected to be length {vid_len}.')

    return output_video, successful


def make_video_array(vid_path, lmrks, w=OUT_IMG_WIDTH, h=OUT_IMG_HEIGHT, dtype=np.uint8):
    vid_len = len(lmrks)
    cap = cv2.VideoCapture(vid_path, cv2.CAP_FFMPEG)
    video_idx = 0
    output_video = np.zeros((vid_len, h, w, 3), dtype=dtype)
    successful = True

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if video_idx == 0:
                img_h, img_w = frame.shape[:2]

            if video_idx < vid_len:
                lmrk = lmrks[video_idx]
            else: #lmrks are shorter than video
                successful = False
                print('ERROR: Fewer landmarks than video frames, must relandmark.')
                break

            lmrk = lmrk.astype(int)
            bbox = get_bbox(lmrk, img_w, img_h)
            square_bbox = get_square_bbox(bbox, img_w, img_h)

            x1,y1,x2,y2 = square_bbox
            cropped = frame[y1:y2, x1:x2]
            if cropped.size < 1:
                resized = np.zeros((h, w, 3), dtype=cropped.dtype)
            else:
                resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
            output_video[video_idx] = resized
            video_idx += 1
        else:
            break

    cap.release()

    if video_idx < vid_len:
        successful = False
        print(f'ERROR: Reached video idx {video_idx} while video was expected to be length {vid_len}.')

    return output_video, successful


def get_bbox(lmrks, img_w, img_h):
    x_min, y_min = lmrks.min(axis=0)
    x_max, y_max = lmrks.max(axis=0)
    x_diff = x_max - x_min
    x_upper_pad = x_diff * 0.05
    x_lower_pad = x_diff * 0.05
    x_min -= x_upper_pad
    x_max += x_lower_pad
    if x_min < 0:
        x_min = 0
    if x_max > img_w:
        x_max = img_w
    y_diff = y_max - y_min
    y_upper_pad = y_diff * 0.3
    y_lower_pad = y_diff * 0.05
    y_min -= y_upper_pad
    y_max += y_lower_pad
    if y_min < 0:
        y_min = 0
    if y_max > img_h:
        y_max = img_h
    bbox = np.array([x_min, y_min, x_max, y_max]).astype(int)
    return bbox


def shift_inside_frame(x1,y1,x2,y2,img_w,img_h):
    if y1 < 0:
        y2 -= y1
        y1 -= y1
    if y2 > img_h:
        shift = y2 - img_h
        y1 -= shift
        y2 -= shift

    if x1 < 0:
            x2 -= x1
            x1 -= x1
    if x2 > img_w:
        shift = x2 - img_w
        x1 -= shift
        x2 -= shift

    return x1,y1,x2,y2


def get_square_bbox(bbox, img_w, img_h):
    x1,y1,x2,y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x1,y1,x2,y2 = shift_inside_frame(x1,y1,x2,y2,img_w,img_h)
    w = x2 - x1
    h = y2 - y1

    ## Push the rectangle out into a square
    if w > h:
        # if w > IN_IMG_HEIGHT:
        #     print('************** Oh no... **************')
        d = w - h
        pad = int(d/2)
        y1 -= pad
        y2 += pad + (d % 2 == 1)
        x1,y1,x2,y2 = shift_inside_frame(x1,y1,x2,y2,img_w,img_h)
    elif w < h:
        # if h > IN_IMG_WIDTH:
        #     print('************** Oh no... **************')
        d = h - w
        pad = int(d/2)
        x1 -= pad
        x2 += pad + (d % 2 == 1)
        x1,y1,x2,y2 = shift_inside_frame(x1,y1,x2,y2,img_w,img_h)

    if x1 < 0:
        x1 = 0
    if x2 > img_w:
        x2 = img_w
    if y1 < 0:
        y1 = 0
    if y2 > img_h:
        y2 = img_h

    w = x2 - x1
    h = y2 - y1
    return int(x1), int(y1), int(x2), int(y2)


import copy
import sys

import numpy as np

import cluster_hue_sat_funcs
import tracking_helpers

sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from multiprocessing import Process, Array, Manager
import cluster_hue_sat_funcs as cluster_funcs
from sklearn.cluster import KMeans
import tensorflow as tf
from collections import deque, Counter, namedtuple
import tracking_helpers as track_helpers
import logging


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

group_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def does_box_overlap(box, other_boxes):
    return


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, color_predictions, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        # color = compute_color_for_labels(id)
        if color_predictions is not None:
            color = group_color[color_predictions[id]]
        else:
            color = (0, 0, 0)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split(os.path.sep)[-1].split('.')[0]
    txt_path = str(Path(out)) + os.path.sep + txt_file_name + '.txt'

    # manager = Manager()
    # lst = manager.list()
    cluster_data = np.empty((0, 32))
    kmeans = None
    color_predictions = None
    collect_color_data = False
    id_class_dict = {key: deque([], maxlen=20) for key in list(range(1, 200))} #Could be better optimized
    id_centroid_dict = {}
    ids_in_last_frame = set()
    ids_to_ignore = set()
    logging.basicConfig(handlers=[logging.FileHandler(filename=r"C:\Users\Andre\PycharmProjects\Yolov5_DeepSort_Pytorch\emerging_boxes.txt",
                                                      encoding='utf-8', mode='w')],
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%F %A %T",
                        level=logging.DEBUG)

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        start_process_time = time.perf_counter()
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confss = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confss.cpu(), im0)

                if len(outputs) > 0:
                    collect_color_data = True
                    bboxes = outputs[:, :4]
                    identities = outputs[:, -1]
                    id_set = set(map(int, identities))

                    if frame_idx == 206 or frame_idx == 11:
                        delete_this_when_notdebug = 0
                    # Check if boxes have split
                    new_ids_to_ignore = ids_to_ignore.copy()
                    id_difference = id_set.symmetric_difference(ids_in_last_frame)  # boxes that have appeared or disappeared
                    new_ids = [id for id in id_difference if id not in id_centroid_dict]

                    # Check if boxes have combined
                    id_lost = [id for id in id_difference if id in id_centroid_dict]  # filter out boxes that have already existed
                    for id in id_lost:
                        closest_box_id = tracking_helpers.find_closest_box(id, identities[identities != id], 125, id_centroid_dict)
                        if closest_box_id is not None:
                            ids_to_ignore.add(tracking_helpers.Id_pair(id, closest_box_id))
                            logging.debug(" At frame index %d, adding ID pair [%d, %d] to the ids to ignore.", frame_idx, id, closest_box_id)

                    for box, id in zip(bboxes, identities):
                        id_centroid_dict[id] = tracking_helpers.xyxy_to_xy_centroid(box)
                    # closest id is the id that remains while, id_pair.id is the one that was lost
                    for id_pair in ids_to_ignore:
                        new_id_box = None
                        if new_ids:
                            new_id_box = tracking_helpers.find_closest_box(id_pair.closest_id, new_ids, 125, id_centroid_dict)
                        if new_id_box is not None:
                            new_ids_to_ignore.remove(tracking_helpers.Id_pair(id_pair.id, id_pair.closest_id))
                            logging.debug(f" At frame index {frame_idx}, Removing ID pair [{id_pair.id}, {id_pair.closest_id}] from the ids to ignore because {new_id_box} appeared too close.")
                        elif id_pair.id in id_set:
                            new_ids_to_ignore.remove(tracking_helpers.Id_pair(id_pair.id, id_pair.closest_id))
                            logging.debug(f" At frame index {frame_idx}, Removing ID pair [{id_pair.id}, {id_pair.closest_id}] from the ids to ignore because {id_pair.id} returned.")
                    ids_to_ignore = new_ids_to_ignore
                    new_ids_to_ignore.clear()


                    #Test
                    # img_pt = copy.deepcopy(im0)
                    # box = bboxes[0, :]
                    # cent = tracking_helpers.xyxy_to_xy_centroid(box)
                    # img_pt = cv2.circle(img_pt, (int(cent[0]), int(cent[1])), radius=5, color=(255, 0, 0), thickness=-1)
                    # cv2.imshow("Centroid", img_pt)
                    # cv2.waitKey(0)
                    #END TEST
                    ids_in_last_frame = set(identities.tolist())
                    hist_data = np.zeros((bboxes.shape[0], 32)) #32 is dependent on window size, number of peaks, etc..
                    hist_data = cluster_funcs.get_histogram_data(im0, bboxes, hist_data)

                # p.join()
                # hist_data = lst[0]
                mode_in_class_id_dict = None

                if frame_idx < 25 and collect_color_data:
                    cluster_data = np.vstack((cluster_data, hist_data))
                elif frame_idx == 25:
                    # start = time.perf_counter()
                    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(cluster_data)
                    # print(f"Time taken to cluster = {time.perf_counter() - start}")
                elif frame_idx > 25:
                    # start = time.perf_counter()
                    color_predictions = kmeans.predict(hist_data)
                    identities = outputs[:, -1] # TODO Change this to ids in last frame to make indices match

                    for predict_class, id in zip(color_predictions, identities):
                        # Filter out any predictions that are labeled as being combined boxes
                        exists_bool_list = [True for x in ids_to_ignore if x.closest_id == int(id)]
                        if not any(exists_bool_list):
                            id_class_dict[id].append(predict_class)
                    mode_in_class_id_dict = cluster_hue_sat_funcs.get_counts(identities, id_class_dict)
                    # print(f"Time taken to predict = {time.perf_counter() - start}")


                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, mode_in_class_id_dict, identities)
                    # to MOT format
                    tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)

                    # Write MOT compliant results to file
                    ids = []
                    if save_txt:
                        for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                            bbox_top = tlwh_bbox[0]
                            bbox_left = tlwh_bbox[1]
                            bbox_w = tlwh_bbox[2]
                            bbox_h = tlwh_bbox[3]
                            identity = output[-1]
                            ids.append(identity)
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
                                                            bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                        # Added by Andrew Hilton
                        #     cluster_save_path = r'C:\Users\Andre\PycharmProjects\Yolov5_DeepSort_Pytorch\feature_vectors.txt'
                        #     with open(cluster_save_path, 'a') as file:
                        #         bboxes = list(map(deepsort._tlwh_to_xyxy, tlwh_bboxs))
                        #         features = deepsort._get_features(bboxes, im0)
                        #         for i in range(len(ids)):
                        #             file.write(str(frame_idx) + ',' + str(ids[i]) + ',' + str(bboxes[i][0]) + ','
                        #                        + str(bboxes[i][1]) + ',' + str(bboxes[i][2]) + ',' + str(bboxes[i][3]) + ','
                        #                        + str(features[i]) + '?;')

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))
            print(f"{s} Done. {t2 - t1 + (time.perf_counter() - start_process_time)}")

            # Stream results
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    # Draw frame number in top left
                cv2.putText(im0, f'frame index = {frame_idx}', (100, 200), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    print(f"Everything still remaining in ids_to_ignore = {ids_to_ignore}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)

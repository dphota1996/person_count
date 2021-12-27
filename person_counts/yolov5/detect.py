import argparse
import time
from pathlib import Path
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import gi
# gi.require_version('Gtk','2.0')
import cv2
import warnings

warnings.filterwarnings('ignore')
import torch
# import itertools
import torch.backends.cudnn as cudnn
from numpy import random
from utils.sort import *

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, time_synchronized


def detect():
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    mot_tracker = Sort()
    save_img = opt.save_img
    source_list = opt.list
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # print(save_dir)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    tf = 1
    person_count = []

    for path, img, im0s, vid_cap, orignal_img, location in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        new_tensor = torch.tensor(pred[0], device='cpu')

        persons_123 = []

        for p in range(len(new_tensor)):
            if int(new_tensor[p][5]) == 0:
                persons_123.append(new_tensor[p])

        if webcam:
            path = path[0]
            im0s = im0s[0]

        # print(im0s.shape)
        # print('img',img.shape)

        p, s, im0, frame = Path(path), '', im0s, getattr(dataset, 'frame', 0)

        if 'person' in source_list:
            if not len(persons_123) == 0:
                # print(persons_123)
                tracked_objects = mot_tracker.update(persons_123)
                tracked_objects[:, :4] = scale_coords(img.shape[2:], tracked_objects[:, :4], im0.shape)  # .round()
                # print(tracked_objects)
                conf_number = 0
                for x1, y1, x2, y2, obj_id, cls_pred in reversed(tracked_objects):
                    conf = persons_123[conf_number][4]
                    conf_number = conf_number + 1
                    c1 = (int(x1), int(y1))
                    c2 = (int(x2), int(y2))
                    line_thickness = 2
                    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
                    color = [random.randint(0, 255) for _ in range(3)]
                    label = f'{names[int(cls_pred)]}'
                    label = '*'
                    # if label=="person":
                    color = [0, 255, 0]
                    # label = label+'_'+str(int(obj_id))+str(float(conf*100))
                    label = label + ' ' + str(round(float(conf * 100), 1)) + '%'
                    # print(person_count)
                    # print(int(obj_id))
                    if not int(obj_id) in person_count:
                        person_count.append(int(obj_id))

                    cv2.rectangle(im0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                    # if not 'person' in label:
                    if label:
                        tf = max(tl - 1, 1)  # font thickness
                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                        cv2.rectangle(im0, c1, c2, color, -1, cv2.LINE_AA)  # filled
                        cv2.putText(im0, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                                    lineType=cv2.LINE_AA)

        orignal_img[int(location[1]):int(location[1] + location[3]),
        int(location[0]):int(location[0] + location[2])] = im0
        # print(im0.shape)
        cv2.rectangle(orignal_img, (int(location[0]), int(location[1])),
                      (int(location[0] + location[2]), int(location[1] + location[3])), [0, 0, 255], thickness=1,
                      lineType=cv2.LINE_AA)
        im0 = orignal_img

        final_list = {}

        if 'person' in source_list:
            final_list['Human'] = str(len(persons_123))

        kl = 0
        text = 'On Screen Count: '
        for key in final_list:
            if kl != 0:
                text = text + '   '
            text = text + key + ': ' + final_list[key] + '   '
            kl = kl + 1

        t_size = cv2.getTextSize(text, 0, fontScale=2 / 3, thickness=tf)[0]

        sizes = im0.shape
        c1 = (30, 50)
        c2 = (t_size[0] + 100, 80)
        startX = c1[0]
        startY = c1[1]
        endX = c2[0]
        endY = c2[1]
        sub_img = im0[startY:endY, startX:endX]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

        cv2.putText(res, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        im0[startY:endY, startX:endX] = res

        if not len(person_count) == 0:
            person_counts = max(person_count)

            final_list = {}
            if 'person' in source_list:
                final_list['Human'] = str(person_counts)

            kl = 0
            text = 'Video Totals:      '
            for key in final_list:
                if kl != 0:
                    text = text + '   '
                text = text + key + ': ' + final_list[key] + '   '
                kl = kl + 1

            t_size = cv2.getTextSize(text, 0, fontScale=2 / 3, thickness=tf)[0]

            # sizes=im0.shape
            c1 = (30, 10)
            c2 = (t_size[0] + 100, 40)
            startX = c1[0]
            startY = c1[1]
            endX = c2[0]
            endY = c2[1]
            sub_img = im0[startY:endY, startX:endX]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

            #cv2.putText(res, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            im0[startY:endY, startX:endX] = res

        print(f'{s}Done. ({t2 - t1:.3f}s)')
        # print(save_path)

        # To show image
        # Stream results
        if view_img:
            cv2.imshow(str(p), im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                # raise StopIteration
                view_img = False
                cv2.destroyAllWindows()
                # if webcam:

                #     break

        # Save results (image with detections)
        if save_img:
            save_path = str(save_dir / p.name)
            print(save_path, p.name)
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)

            else:  # 'video'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    fp = os.path.splitext(vid_path)
                    vid_path111 = fp[0] + '.avi'
                    # print(vid_path111)
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(vid_path111, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(im0)

    vid_cap.release()
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    cv2.destroyAllWindows()
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='last.pt', help='model.pt path(s)')
    parser.add_argument("--list", nargs="+", default=["person"])
    parser.add_argument('--save_img', type=bool, default=True, help='write video or not')
    parser.add_argument('--source', type=str, default='C:/Users/user/Downloads/person_counts/person_counts/yolov5/test_video.mp4',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.30, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True, action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs=1, type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', default=False,
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

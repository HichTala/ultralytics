import os

import cv2
import numpy as np
import torch
from timm.models import create_model
from torchvision import transforms

import utils
from ultralytics import YOLO


def get_lower_edges(coordinates, h, w):
    # Convert the coordinates to a numpy array if it's not already
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    try:
        set1 = coordinates[np.argsort(coordinates[:, 0])][:2]
        set2 = coordinates[np.argsort(coordinates[:, 1])][:2]

        nrows, ncols = set1.shape
        dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                 'formats': ncols * [set1.dtype]}

        intersection = np.intersect1d(set1.view(dtype), set2.view(dtype))
        intersection = intersection.view(set1.dtype).reshape(-1, ncols)[0]

        edge1 = np.where((coordinates == intersection).all(axis=1))[0][0]

        set1 = coordinates[np.argsort(coordinates[:, 0])][:2]
        set2 = coordinates[np.argsort(coordinates[:, 1])][2:]

        nrows, ncols = set1.shape
        dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                 'formats': ncols * [set1.dtype]}

        intersection = np.intersect1d(set1.view(dtype), set2.view(dtype))
        intersection = intersection.view(set1.dtype).reshape(-1, ncols)[0]

        edge2 = np.where((coordinates == intersection).all(axis=1))[0][0]

        set1 = coordinates[np.argsort(coordinates[:, 0])][2:]
        set2 = coordinates[np.argsort(coordinates[:, 1])][:2]

        nrows, ncols = set1.shape
        dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                 'formats': ncols * [set1.dtype]}

        intersection = np.intersect1d(set1.view(dtype), set2.view(dtype))
        intersection = intersection.view(set1.dtype).reshape(-1, ncols)[0]

        edge3 = np.where((coordinates == intersection).all(axis=1))[0][0]

        set1 = coordinates[np.argsort(coordinates[:, 0])][2:]
        set2 = coordinates[np.argsort(coordinates[:, 1])][2:]

        nrows, ncols = set1.shape
        dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                 'formats': ncols * [set1.dtype]}

        intersection = np.intersect1d(set1.view(dtype), set2.view(dtype))
        intersection = intersection.view(set1.dtype).reshape(-1, ncols)[0]

        edge4 = np.where((coordinates == intersection).all(axis=1))[0][0]

        if coordinates[edge1][0] - coordinates[edge3][0] < coordinates[edge1][1] - coordinates[edge2][1]:
            if edge1 < h:
                return [coordinates[edge1], coordinates[edge2]], [coordinates[edge3], coordinates[edge4]]
            else:
                return [coordinates[edge2], coordinates[edge1]], [coordinates[edge4], coordinates[edge3]]
        else:
            if edge1 < w:
                return [coordinates[edge1], coordinates[edge3]], [coordinates[edge2], coordinates[edge4]]
            else:
                return [coordinates[edge3], coordinates[edge1]], [coordinates[edge4], coordinates[edge2]]

    except:
        return None, None


def crop_min_area_rect(img, box, angle):
    # rotate img
    # angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    # rotate bounding box
    # rect0 = (rect[0], rect[1], 0.0)
    # box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0
    # cv2.drawContours(img_rot, [pts], 0, (255, 152, 119), 2)

    # crop
    img_crop = img_rot[pts[:, 1].min():pts[:, 1].max(), pts[:, 0].min():pts[:, 0].max()]
    return img_crop


with open("test.ydk") as f:
    ydk = [line.rstrip() for line in f.readlines()]

directory = '/home/cose-ia/yugioh-card-classification/cardDatabaseFormatted/monster/'

classes = [d.name for d in os.scandir(directory) if d.is_dir()]
classes.sort()
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

deck_card_id = []

for card_id in ydk:
    if card_id[0] in '0123456789':
        for card_name in class_to_idx.keys():
            if card_id[0] == '0':
                if card_id[1:] in card_name:
                    deck_card_id.append(class_to_idx[card_name])
            if card_id in card_name:
                deck_card_id.append(class_to_idx[card_name])

deck_card_id = list(set(deck_card_id))

model_classification = create_model(
    'beit_base_patch16_224',
    pretrained=False,
    num_classes=4602,
    drop_rate=0,
    drop_path_rate=0.1,
    attn_drop_rate=0,
    drop_block_rate=None,
    use_rel_pos_bias=True,
    use_abs_pos_emb=False,
    init_values=0.1,
)
checkpoint = torch.load('/home/cose-ia/beit/runs/monster/checkpoint-best.pth')
utils.load_state_dict(model_classification, checkpoint['model'])

model_regression = YOLO("./runs/detect/train14/weights/best.pt")

results = model_regression(
    source="/home/cose-ia/Downloads/YuGiOh_YOLO.v2i.yolov8/video/2023-07-02 16-26-49 (online-video-cutter.com).mp4",
    show_labels=False,
    save=False
)

colors = {
    'spell': np.array([120, 130, 40]),
    'monster': np.array([81, 86, 138]),
    'link': np.array([97, 66, 8]),
    'xyz': np.array([61, 46, 21]),
    'syncro': np.array([180, 173, 165]),
    'trap': np.array([124, 77, 133])
}

card_types = ['spell', 'monster', 'link', 'xyz', 'syncro', 'trap']

tol = {
    'spell': 35,
    'monster': 35,
    'link': 35,
    'xyz': 20,
    'syncro': 35,
    'trap': 35,
}

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter('filename.avi', fourcc, 30, (1280, 720))

try:
    for frame, result in enumerate(results[70:]):
        print(frame)
        img = result.orig_img.copy()
        for nbox, boxes in enumerate(result):
            x1, y1, x2, y2, = map(int, boxes.boxes.xyxy.squeeze())
            roi = img[y1:y2, x1:x2].copy()
            roi_original = img[y1:y2, x1:x2]

            width = x2 - x1
            height = y2 - y1

            ones1 = np.ones(
                (int(0.71 * height) - int(0.18 * height), width - int(0.115 * width) - int(0.115 * width), 3)
            )

            ones2 = np.ones(
                (int(0.27 * height) - int(0.1 * height), width - int(0.115 * width) - int(0.115 * width), 3)
            )

            roi[
            height - int(0.71 * height): height - int(0.18 * height),
            int(0.115 * width):width - int(0.115 * width)
            ] = ones1 * [1000, 1000, 1000]

            roi[
            int(0.1 * height):int(0.27 * height),
            int(0.115 * width):width - int(0.115 * width)
            ] = ones2 * [1000, 1000, 1000]

            roi = roi.reshape((roi.shape[0] * roi.shape[1], 3)).astype(float)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 152, 119), thickness=2)

            counts = [
                np.isclose(
                    np.array([np.linalg.norm(diff) for diff in (roi - colors[card_type])]),
                    np.zeros(roi.shape[0]),
                    atol=tol[card_type]
                ).sum()
                for card_type in card_types
            ]

            index_max = np.argmax(counts)
            card_type = card_types[index_max]

            cv2.putText(img, card_type, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)

            gray = cv2.cvtColor(roi_original, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            equalized = cv2.equalizeHist(gray)
            _, thresh = cv2.threshold(equalized, 140, 255, cv2.THRESH_BINARY)

            kernel = np.ones((7, 7), np.uint8)
            edged = cv2.erode(thresh, kernel, iterations=3)
            edged = cv2.dilate(edged, kernel, iterations=3)

            cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            min_area = 2000
            max_area = 4000

            c = cnts[np.array(list(map(cv2.contourArea, cnts))).argmax()]

            area = cv2.contourArea(c)
            if min_area < area < max_area:
                print(area)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(roi_original, [box], 0, (152, 255, 119), 2)
                edges1, edges2 = get_lower_edges(box, roi_original.shape[0], roi_original.shape[1])
                if edges1 is None:
                    break

                dx1 = edges1[1][0] - edges1[0][0]
                dx2 = edges2[1][0] - edges2[0][0]
                dy1 = edges1[1][1] - edges1[0][1]
                dy2 = edges2[1][1] - edges2[0][1]

                dst1 = (dx1 ** 2 + dy1 ** 2) ** 0.5
                dst2 = (dx2 ** 2 + dy2 ** 2) ** 0.5

                box_artwork = np.array([
                    [edges1[1][0] + dx1 * 5 / dst1, edges1[1][1] + dy1 * 5 / dst1],
                    [edges2[1][0] + dx2 * 5 / dst2, edges2[1][1] + dy2 * 5 / dst2],
                    [edges2[1][0] + dx2 * 95 / dst2, edges2[1][1] + dy2 * 100 / dst2],
                    [edges1[1][0] + dx1 * 95 / dst1, edges1[1][1] + dy1 * 100 / dst1],
                ], dtype=np.int64)

                contour2 = np.intp(cv2.boxPoints(cv2.minAreaRect(box_artwork)))
                # cv2.drawContours(roi_original, [box_artwork], 0, (119, 152, 255), 2)
                artwork = crop_min_area_rect(roi_original, box_artwork,
                                             180 - np.arctan(dx1 / dy1) * 180 / np.pi)

                # cv2.drawContours(roi_original, [contour2], 0, (255, 152, 119), 2)
                if artwork.shape[0] != 0 and artwork.shape[1] != 0:

                    image = transforms.ToTensor()(artwork)
                    image = transforms.Resize((224, 224), antialias=True)(image)
                    image = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image)
                    model_classification.eval()
                    output = model_classification(image.unsqueeze(0))

                    _, indices = torch.sort(output, descending=True)

                    for k, i in enumerate(indices[0]):
                        if i in deck_card_id:
                            cv2.imwrite(
                                '/home/cose-ia/Downloads/YuGiOh_YOLO.v2i.yolov8/ROI/video-remote-artwork-1/frame_{}_box_{}_{}.png'.format(
                                    frame, nbox, classes[i]),
                                artwork
                            )
                            print(classes[i])
                            print(k)
                            break

        # cv2.putText(img, str(i), (10, 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        video_writer.write(img)
except KeyboardInterrupt:
    video_writer.release()

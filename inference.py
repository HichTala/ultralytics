import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import binascii

import scipy
import scipy.misc
import scipy.cluster
from ultralytics import YOLO

model = YOLO("./runs/YOLO_YuGiOh_300eps_300523/train14/weights/best.pt")

# metrics = model.val()  # evaluate model performance on the validation set
results = model(
    source="./Videos/2023-07-02 16-26-49 (online-video-cutter.com).mp4",
    show_labels=False,
    save=False
)

clusters = 10

kmeans = KMeans(n_clusters=clusters, n_init='auto')

# colors = {
#     'spell': np.array([75, 120, 120]),
#     'monster': np.array([164, 113, 38]),
#     'link': np.array([38, 38, 165]),
#     'xyz': np.array([0, 0, 0]),
#     'syncro': np.array([150, 150, 175]),
# }

colors = [
    np.array([0, 113, 112]),
    np.array([115, 76, 63]),
    np.array([51, 112, 150]),
    np.array([25, 49, 68]),
    np.array([165, 173, 180]),
]

card_types = ['spell', 'monster', 'link', 'xyz', 'syncro']
print(results[0].orig_shape)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter('filename.avi', fourcc, 30, (1280, 720))

try:
    for i, result in enumerate(results[0:]):
        print(i)
        img = result.orig_img.copy()
        for boxes in result:
            x1, y1, x2, y2, = map(int, boxes.boxes.xyxy.squeeze())
            roi = img[y1:y2, x1:x2]
            roi = roi.reshape((roi.shape[0] * roi.shape[1], 3)).astype(float)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

            # kmeans.fit(roi)

            # codes = kmeans.cluster_centers_
            # print('cluster centres:\n', codes)

            counts = [
                np.isclose(
                    np.array([np.linalg.norm(diff) for diff in (roi - color)]),
                    np.zeros(roi.shape[0]),
                    atol=35
                ).sum()
                for color in colors
            ]

            index_max = np.argmax(counts)  # find most frequent
            card_type = card_types[index_max]
            
            # if len(result) > 1:
            #     breakpoint()
            # min = 500
            # label = ''
            # for cluster in kmeans.cluster_centers_:
            # if len(result) > 1:
            #     print("cluster", cluster, x1, y1, x2, y2)
            # for card_type, color in colors.items():
            #     distance = np.linalg.norm(colour - color)
            #     # if len(result) > 1:
            #     #     print(distance, card_type)
            #     if distance < min:
            #         min = distance
            #         label = card_type

            cv2.putText(img, card_type, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        video_writer.write(img)
        # if len(result) > 1:
        #     break
        # print(img.shape)
        # cv2.imshow("", img)
        # cv2.waitKey()
except KeyboardInterrupt:
    video_writer.release()

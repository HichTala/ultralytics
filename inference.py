import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

from ultralytics import YOLO

model = YOLO("./runs/detect/train14/weights/best.pt")

# metrics = model.val()  # evaluate model performance on the validation set
results = model(
    source="/home/cose-ia/Downloads/YuGiOh_YOLO.v2i.yolov8/video/2023-07-02 16-38-07.mp4",
    show_labels=False,
    save=False
)

clusters = 5

kmeans = KMeans(n_clusters=clusters, n_init='auto')

colors = {
    'spell': np.array([75, 120, 120]),
    'trap': np.array([110, 50, 90]),
    'monster': np.array([164, 113, 38]),
    'link': np.array([38, 38, 165]),
}
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
video_writer = cv2.VideoWriter('filename.mp4', fourcc, 10, results[0].orig_shape, 3)



for i, result in enumerate(results):
    print('{}/690'.format(i))
    img = result.orig_img.copy()
    for boxes in result:
        x1, y1, x2, y2, = map(int, boxes.boxes.xyxy.squeeze())
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

        roi = img[y1:y2, x1:x2]
        roi = roi.reshape((roi.shape[0] * roi.shape[1], 3))
        kmeans.fit(roi)

        min = 500
        label = ''
        for cluster in kmeans.cluster_centers_:
            for card_type, color in colors.items():
                distance = np.linalg.norm(cluster - color)
                if distance < min:
                    min = distance
                    label = card_type

        cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)

    print(img.shape)
    video_writer.write(img)
    cv2.imshow("", img)
    cv2.waitKey()
video_writer.release()

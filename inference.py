import cv2
import numpy as np

from ultralytics import YOLO

model = YOLO("./runs/detect/train14/weights/best.pt")

results = model(
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
    for i, result in enumerate(results[0:]):
        print(i)
        img = result.orig_img.copy()
        for boxes in result:
            x1, y1, x2, y2, = map(int, boxes.boxes.xyxy.squeeze())
            roi = img[y1:y2, x1:x2].copy()

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
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

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
        # cv2.putText(img, str(i), (10, 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        video_writer.write(img)
except KeyboardInterrupt:
    video_writer.release()

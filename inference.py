import cv2

from ultralytics import YOLO

model = YOLO("./runs/detect/train14/weights/best.pt")

# metrics = model.val()  # evaluate model performance on the validation set
results = model(
    source="/home/cose-ia/Downloads/YuGiOh_YOLO.v2i.yolov8/video/video-2.mp4",
    show_labels=False,
    save=False
)
# breakpoint()

frame = 0
for result in results:
    box = 0
    for boxes in result:
        x1, y1, x2, y2, = map(int, boxes.boxes.xyxy.squeeze())
        roi = result.orig_img[y1:y2, x1:x2]
        try:
            cv2.imwrite(
                '/home/cose-ia/Downloads/YuGiOh_YOLO.v2i.yolov8/ROI/video-2/frame_{}_box_{}.png'.format(frame, box),
                roi
            )
        except:
            breakpoint()
        box += 1
    frame += 1

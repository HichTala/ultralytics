import cv2
import numpy as np

from ultralytics import YOLO


def get_lower_edges(coordinates):
    # Convert the coordinates to a numpy array if it's not already
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)

    # Sort the coordinates in ascending order based on the y-coordinate
    sorted_coordinates = coordinates[np.argsort(coordinates[:, 1])]

    # Get the two lower vertices of the rectangle
    lower_vertices = sorted_coordinates[:2]

    # Calculate the center point of the rectangle
    center = np.mean(coordinates, axis=0)

    # Sort the two lower vertices based on their angle with respect to the center
    sorted_lower_vertices = sorted(lower_vertices,
                                   key=lambda coord: np.arctan2(coord[1] - center[1], coord[0] - center[0]))

    # Calculate the other two vertices
    upper_vertices = sorted_coordinates[2:]

    # Sort the upper vertices based on their angle with respect to the center
    sorted_upper_vertices = sorted(upper_vertices,
                                   key=lambda coord: np.arctan2(coord[1] - center[1], coord[0] - center[0]))

    # Return the two pairs of coordinates of the two lower edges
    lower_edge1 = sorted_lower_vertices[0], sorted_upper_vertices[1]
    lower_edge2 = sorted_lower_vertices[1], sorted_upper_vertices[0]

    return lower_edge1, lower_edge2


model = YOLO("./runs/YOLO_YuGiOh_300eps_300523/train14/weights/best.pt")

results = model(
    source="./Videos/2023-07-02 16-26-49 (online-video-cutter.com).mp4",
    show_labels=False,
    save=False,
    device='cpu'
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
            max_area = 8000
            for c in cnts:
                area = cv2.contourArea(c)
                if min_area < area < max_area:
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    edges1, edges2 = get_lower_edges(box)

                    dx1 = abs(edges1[0][0] - edges1[1][0])
                    dx2 = abs(edges2[0][0] - edges2[1][0])
                    dy1 = abs(edges1[0][1] - edges1[1][1])
                    dy2 = abs(edges2[0][1] - edges2[1][1])

                    dst1 = (dx1 ** 2 + dy1 ** 2) ** 0.5
                    dst2 = (dx2 ** 2 + dy2 ** 2) ** 0.5

                    box_artwork = np.array([
                        [edges1[1][0] + dx1 * 5 / dst1, edges1[1][1] + dy1 * 5 / dst1],
                        [edges2[1][0] + dx2 * 5 / dst2, edges2[1][1] + dy2 * 5 / dst2],
                        [edges2[1][0] + dx2 * 100 / dst2, edges2[1][1] + dy2 * 100 / dst2],
                        [edges1[1][0] + dx1 * 100 / dst1, edges1[1][1] + dy1 * 100 / dst1],
                    ], dtype=np.int64)

                    cv2.drawContours(roi_original, [box_artwork], 0, (119, 152, 255), 2)

        # cv2.putText(img, str(i), (10, 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        video_writer.write(img)
except KeyboardInterrupt:
    video_writer.release()

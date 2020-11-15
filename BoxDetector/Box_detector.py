import cv2

import numpy as np
import pupil_apriltags as apriltag


class BoxDetector:
    """Пример использования:
    detector = BoxDetector()
    # Детекция тегов
    detector.detections(frame)
    # Получить значения координат
    coords = detector.get_values()
    """

    def __init__(self):
        """Конструктор класса"""
        self.tag_detector = apriltag.Detector(
            families="tagStandard41h12",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        self.detect_dict = {1: 0, 2: 0, 3: 0}

    def detections(self, frame):
        """Детектит теги на фрейме
        Ключевой аргумент:
        frame -- фрейм из видеопотока файла / камеры
        """
        self.color_frame = frame
        self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.detection = self.tag_detector.detect(self.gray_frame)

    def _selecting_by_square(self):
        """сортирует теги по площади и выдает теги , площадь которых (для каждого id) максимальна"""
        self.detect_list = []

        first_corners = []
        sq1 = []

        second_corners = []
        sq2 = []

        third_corners = []
        sq3 = []
        first_idx = 0
        second_idx = 0
        third_idx = 0
        if len(self.detection) != 0:
            for i in range(1, 4):
                for detect in self.detection:
                    if detect.tag_id == i:
                        self.detect_list.append([i, detect.corners])

            for i, corners in self.detect_list:
                if i == 1:
                    first_corners.append(corners)
                if i == 2:
                    second_corners.append(corners)
                if i == 3:
                    third_corners.append(corners)
            for corn in first_corners:
                x1_left_Up = corn[0][0]
                y1_left_Up = corn[0][1]

                x1_righ_bot = corn[2][0]
                y1_righ_bot = corn[2][1]
                square = (x1_righ_bot - x1_left_Up) * (y1_righ_bot - y1_left_Up)
                sq1.append(square)
            for corn in second_corners:
                x1_left_Up = corn[0][0]
                y1_left_Up = corn[0][1]

                x1_righ_bot = corn[2][0]
                y1_righ_bot = corn[2][1]
                square = (x1_righ_bot - x1_left_Up) * (y1_righ_bot - y1_left_Up)
                sq2.append(square)
            for corn in third_corners:
                x1_left_Up = corn[0][0]
                y1_left_Up = corn[0][1]

                x1_righ_bot = corn[2][0]
                y1_righ_bot = corn[2][1]
                square = (x1_righ_bot - x1_left_Up) * (y1_righ_bot - y1_left_Up)
                sq3.append(square)

            try:
                first_idx = sq1.index(max(sq1))
                self.detect_dict[1] = first_corners[first_idx]
            except:
                self.detect_dict[1] = 0
            try:
                second_idx = sq2.index(max(sq2))
                self.detect_dict[2] = second_corners[second_idx]
            except:
                self.detect_dict[2] = 0
            try:
                third_idx = sq3.index(max(sq3))
                self.detect_dict[3] = third_corners[third_idx]
            except:
                self.detect_dict[3] = 0

    def get_values(self):
        """Возвращает координаты тегов
        координаты возвращаются в следующем формате:
        {id:<num>, bbox:(x_left, y_top, x_right, y_bottom)}
        <num> натуральное число от 1 до 3
        координаты - вещественные числа в диапазоне [0;1]
        """
        self._selecting_by_square()
        height, width, _ = self.color_frame.shape
        try:
            return_dict = {}
            for i, val in zip(self.detect_dict, self.detect_dict.values()):
                if type(val) == int:
                    continue
                val[1] = [val[1][0] / width, val[1][1] / height]
                val[3] = [val[3][0] / width, val[3][1] / height]
                return_dict[i] = [val[3][0], val[3][1], val[1][0], val[1][1]]
            return return_dict
        except Exception as e:
            print(e)
            print("no tags found")
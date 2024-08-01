import cv2
import sys
import random
import sqlite3
import mediapipe as mp
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog


class JumpAlgorithm:
    def __init__(self, cv_capture, video_label):
        # 获取UI中读取的视频和显示区域
        self.cv_capture = cv_capture
        self.video_label = video_label
        # 为当前视频检测对象生成随机英文名
        self.random_id = random.randint(100000, 999999)
        self.random_name = self.generate_random_name()
        self.jump_round = 1
        self.white_line_coords = None
        self.jump_distance_recoder = []

    # 随机生成名字
    def generate_random_name(self):
        first_names = ["John", "Jane", "Alex", "Emily", "Chris", "Katie", "Michael", "Sarah"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
        return f"{random.choice(first_names)} {random.choice(last_names)}"

    # 算法主函数，包含主要的处理逻辑
    def run_algorithm(self):
        # 使用mediapipe中的pose工具处理图像
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                            min_detection_confidence=0.5)
        jump_distances = []
        toe_positions = []
        heel_positions = []
        window_size = 3  # 时间窗口大小
        threshold_static = 0.002  # 静止状态阈值
        threshold_dynamic = 0.1  # 起跳状态阈值
        fall_threshold = 0.05  # 跌倒判断阈值
        distance_threshold = 0.01  # 用于比较当前右手位置与上一次终点位置的距离阈值
        previous_point = []
        start_point = None
        end_point = None
        jump_started = False

        while self.cv_capture.isOpened():
            ret, frame = self.cv_capture.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # 提取右脚脚尖、右脚脚后跟和右手的数据
                right_toe = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
                right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
                right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                toe_positions.append(right_toe)
                heel_positions.append(right_heel)
                if len(toe_positions) > window_size:
                    toe_positions.pop(0)
                if len(heel_positions) > window_size:
                    heel_positions.pop(0)

                # 跌倒检测,检测生效时，上一次的成绩作废，重新计算最终成绩
                if self.is_falling(right_hand, right_toe, fall_threshold) and previous_point:
                    # 判断此时右脚位置与上一次终点位置的距离，避免其他摔倒影响成绩计算
                    distance_to_previous_end = np.sqrt((right_heel.x - previous_point[1].x) ** 2 +
                                                       (right_heel.y - previous_point[1].y) ** 2)
                    # 利用脚步位置变化确定还在成绩判定范围内时，计算新的分数并更新
                    if distance_to_previous_end < distance_threshold:
                        previous_point[1] = right_hand
                        distance = np.sqrt((previous_point[0].x - previous_point[1].x) ** 2 +
                                           (previous_point[0].y - previous_point[1].y) ** 2) * 5
                        if distance < jump_distances[-1]:
                            jump_distances[-1] = distance
                            self.jump_distance_recoder[-1] = f'  R{self.jump_round - 1}[FALL]:{distance:.2f}m '
                            start_point = None
                            end_point = None

                # 计算滑动窗口内右脚脚尖坐标的变化量，由此判断起点终点和起跳状态
                if len(toe_positions) == window_size:
                    deltas = [np.sqrt((toe_positions[i].x - toe_positions[i - 1].x) ** 2 +
                                      (toe_positions[i].y - toe_positions[i - 1].y) ** 2)
                              for i in range(1, window_size)]
                    avg_delta = np.mean(deltas)
                    if avg_delta < threshold_static and not jump_started:
                        start_point = toe_positions[0]
                    if avg_delta > threshold_dynamic and not jump_started and start_point:
                        jump_started = True
                    if avg_delta < threshold_static and jump_started:
                        end_point = heel_positions[-1]
                        jump_started = False
                        if start_point and end_point:
                            previous_point = [start_point, end_point]
                            distance = np.sqrt((start_point.x - end_point.x) ** 2 +
                                               (start_point.y - end_point.y) ** 2) * 5
                            jump_distances.append(distance)
                            self.jump_distance_recoder.append(f'R{self.jump_round}:{distance:.2f}m ')
                            self.jump_round += 1
                            start_point = None
                            end_point = None

                # 若处于起跳状态则检查是否踩线
                if not self.white_line_coords and start_point:
                    frame = self.detect_white_line(frame, start_point)
                elif start_point:
                    frame = self.detect_treading_ine(frame, right_toe, right_heel)
                # 结束本轮处理，更新结果
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                display_text = f'[{self.random_name}]   {"".join(self.jump_distance_recoder)}'
                cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # 更新video_label上的帧
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            qt_image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
            QApplication.processEvents()

        self.cv_capture.release()
        cv2.destroyAllWindows()
        recoder = self.process_and_save(jump_distances)
        return recoder

    def is_falling(self, right_hand, right_toe, fall_threshold):
        vertical_distance = abs(right_hand.y - right_toe.y)
        return vertical_distance < fall_threshold

    def detect_treading_ine(self, frame, right_toe, right_heel):
        cv2.line(frame, self.white_line_coords[0], self.white_line_coords[1], (0, 255, 0), 2)
        right_toe_pos = (int(right_toe.x * frame.shape[1]), int(right_toe.y * frame.shape[0]))
        right_heel_pos = (int(right_heel.x * frame.shape[1]), int(right_heel.y * frame.shape[0]))

        # 检测两条线段是否相交
        def line_intersection(p0, p1, p2, p3):
            v0 = (p1[0] - p0[0], p1[1] - p0[1])
            v1 = (p3[0] - p2[0], p3[1] - p2[1])
            denom = v0[0] * v1[1] - v0[1] * v1[0]
            if denom == 0:
                return False
            else:
                t = ((p2[0] - p0[0]) * v1[1] - (p2[1] - p0[1]) * v1[0]) / denom
                u = -((p0[0] - p2[0]) * v0[1] - (p0[1] - p2[1]) * v0[0]) / denom
                return 0 <= t <= 1 and 0 <= u <= 1

        intersection = line_intersection(right_toe_pos, right_heel_pos,
                                         self.white_line_coords[0], self.white_line_coords[1])
        if intersection:
            text = 'Treading Line Warning!!!!'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_w, text_h = text_size
            img_h, img_w, _ = frame.shape
            x = (img_w - text_w) // 2
            y = (img_h + text_h) // 2
            cv2.rectangle(frame, (x - 10, y - text_h - 10), (x + text_w + 10, y + 10), (0, 0, 0),
                          cv2.FILLED)
            cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
        return frame

    # 检测跳远起点直线位置并保存，用于判定是否存在踩线行为
    def detect_white_line(self, frame, start_point, min_dist_to_line=20):
        # 检测图像中的直线，获取起跳点直线坐标
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.bitwise_and(mask_white, cv2.bitwise_not(mask_red))
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
        # 绘制检测到的直线，并记录白线坐标
        if lines is not None:
            # min_dist = 9999
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if start_point:
                    # Calculate the center point of the detected line
                    line_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    start_center = (int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0]))
                    # Calculate the Euclidean distance between the start point and the line center
                    dist_to_line = np.sqrt(
                        (start_center[0] - line_center[0]) ** 2 + (start_center[1] - line_center[1]) ** 2)
                    if dist_to_line < min_dist_to_line:  # 可调整阈值
                        self.white_line_coords = ((x1, y1), (x2, y2))
                        # break
            #         if dist_to_line < min_dist:
            #             min_dist = dist_to_line
            # print(min_dist)
        return frame

    # 处理计算后的数据并保存至SQLite数据库
    def process_and_save(self, distances):
        recoder_score = {
            'id': self.random_id,
            'p_name': self.random_name,
            'score': f"[MAX]:{max(distances):.2f}m [Record]: {', '.join(self.jump_distance_recoder)}",
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        db = sqlite3.connect('jump_records.db')
        cursor = db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jump_records (
                id INTEGER PRIMARY KEY, 
                p_name TEXT, 
                score TEXT, 
                update_at TEXT, 
                created_at TEXT
            )
        """)
        cursor.execute("""
            INSERT INTO jump_records (id, p_name, score, update_at, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (recoder_score['id'], recoder_score['p_name'], recoder_score['score'],
              datetime.now().strftime('%Y-%m-%d %H:%M:%S'), recoder_score['created_at']))
        db.commit()
        cursor.close()
        db.close()
        recoder = f"·Time：{recoder_score['created_at']} \n" \
                  f"·Info: {recoder_score['p_name']} ({recoder_score['id']})\n" \
                  f"·Score：{recoder_score['score']}"
        return recoder


class JumpDistanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.video_path = None
        self.cv_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    # 初始化显示界面
    def init_ui(self):
        # 设置基本界面
        self.setWindowTitle('立定跳远检测')
        self.setGeometry(100, 100, 800, 600)
        # 视频显示区域
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.set_default_image()
        # 成绩记录区域
        self.text_label = QLabel('', self)
        # 功能控件区域
        hbox = QHBoxLayout()
        self.btn_open = QPushButton('打开视频', self)
        self.btn_open.clicked.connect(self.open_video)
        self.btn_start = QPushButton('开始检测', self)
        self.btn_start.clicked.connect(self.start_detection)
        hbox.addWidget(self.btn_open)
        hbox.addWidget(self.btn_start)
        # 设置布局展示器
        container = QVBoxLayout()
        container.addWidget(self.video_label)
        container.addWidget(self.text_label)
        container.addLayout(hbox)
        self.setLayout(container)

    # 初始化视频显示区域
    def set_default_image(self):
        width, height = 800, 450
        image = Image.new('RGB', (width, height), color=(211, 211, 211))  # 灰色背景
        draw = ImageDraw.Draw(image)
        text = "Please select the video to be tested......"
        font = ImageFont.truetype("arial.ttf", 20, encoding='utf-8')
        text_width, text_height = draw.textsize(text, font=font)
        position = ((width - text_width) // 2, (height - text_height) // 2)
        draw.text(position, text, fill="black", font=font)
        image = image.convert("RGB")
        data = image.tobytes("raw", "RGB")
        q_image = QImage(data, image.width, image.height, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    # 选择视频文件
    def open_video(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi);;所有文件 (*)",
                                                  options=options)
        if fileName:
            self.text_label.setText("")
            self.timer.stop()
            self.video_path = fileName
            self.cv_capture = cv2.VideoCapture(self.video_path)
            self.timer.start(30)

    # 根据时间的变化显示内容，若无视频则显示默认画面
    def update_frame(self):
        ret, frame = self.cv_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        else:
            self.timer.stop()
            self.set_default_image()

    # 点击开始检测按钮后暂停原始画面替换为目标检测画面
    def start_detection(self):
        if self.video_path:
            self.timer.stop()
            jump_detection = JumpAlgorithm(self.cv_capture, self.video_label)
            recoder = jump_detection.run_algorithm()
            self.text_label.setText(recoder)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = JumpDistanceApp()
    ex.show()
    sys.exit(app.exec_())

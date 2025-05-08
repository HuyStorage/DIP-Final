import csv
import copy
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque
import tensorflow as tf
import streamlit as st
import time
import datetime
import os
from static.utils import *


class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded


class KeyPointClassifier(object):
    def __init__(self, num_threads=1):
        model_name = "keypoint_classifier.tflite"
        model_path = get_file_path(model_name)
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path, num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        input_details_tensor_index = self.input_details[0]["index"]
        self.interpreter.set_tensor(
            input_details_tensor_index, np.array([landmark_list], dtype=np.float32)
        )
        self.interpreter.invoke()
        output_details_tensor_index = self.output_details[0]["index"]
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))
        return result_index


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[3]),
            (0, 255, 0),
            3,
        )
        cv.line(
            image,
            tuple(landmark_point[3]),
            tuple(landmark_point[4]),
            (0, 255, 0),
            3,
        )

        # Index finger
        cv.line(
            image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 255, 255), 3
        )
        cv.line(
            image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 255, 255), 3
        )
        cv.line(
            image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 255, 255), 3
        )

        # Middle finger
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 255), 3
        )
        cv.line(
            image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 255), 3
        )
        cv.line(
            image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 255), 3
        )

        # Ring finger
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[14]),
            (255, 0, 255),
            3,
        )
        cv.line(
            image,
            tuple(landmark_point[14]),
            tuple(landmark_point[15]),
            (255, 0, 255),
            3,
        )
        cv.line(
            image,
            tuple(landmark_point[15]),
            tuple(landmark_point[16]),
            (255, 0, 255),
            3,
        )

        # Little finger
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[18]),
            (0, 165, 255),
            3,
        )
        cv.line(
            image,
            tuple(landmark_point[18]),
            tuple(landmark_point[19]),
            (0, 165, 255),
            3,
        )
        cv.line(
            image,
            tuple(landmark_point[19]),
            tuple(landmark_point[20]),
            (0, 165, 255),
            3,
        )

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[0]),
            tuple(landmark_point[1]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[1]),
            tuple(landmark_point[2]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[5]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[9]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[13]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[17]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[0]),
            (255, 255, 255),
            2,
        )

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + " (" + hand_sign_text + ")"
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    return image


def draw_info(image, fps):
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    return image


@st.cache_resource(show_spinner=False)
def load_keypoint_classifier():
    return KeyPointClassifier()


def app():
    st.markdown("""
            <div class="center-text">
                <h2>✨ <span style="background: linear-gradient(90deg, #3f51b5, #2196f3);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-weight: bold;
                    margin-bottom: 20px;">Nhận dạng cử chỉ tay</span>
                </h2>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="center-text" style="margin-bottom: 15px">Ứng dụng sử dụng mô hình học sâu để nhận dạng cử chỉ tay theo thời gian thực.</div>', unsafe_allow_html=True)
    
    img_containter = st.empty()
    countdown_placeholder = st.empty() 

    if "detecting" not in st.session_state:
        st.session_state.detecting = False

    if not st.session_state.detecting:
        if st.button("▶️ Bắt đầu nhận dạng"):
            st.session_state.detecting = True
            st.rerun()
        if "captured_images" in st.session_state:
            st.markdown("### 🖼️ Các ảnh đã chụp:")
            for idx, img in enumerate(st.session_state.captured_images):
                st.image(img, caption=f"Ảnh {idx + 1}", channels="BGR", use_container_width=True)
    else:
        if st.button("🛑 Dừng nhận dạng"):
            st.session_state.detecting = False
            st.rerun()

    if st.session_state.detecting:

        cap = cv.VideoCapture(0)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        keypoint_classifier = load_keypoint_classifier()

        label_file_name = "keypoint_classifier_label.csv"
        label_path = get_file_path(label_file_name)
        with open(label_path, encoding="utf-8-sig") as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

        cvFpsCalc = CvFpsCalc(buffer_len=10)

        while cap.isOpened():
            fps = cvFpsCalc.get()

            _, image = cap.read()
            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    label = keypoint_classifier_labels[hand_sign_id]

                    debug_image = draw_bounding_rect(debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(debug_image, brect, handedness, label)

                    if label == "Say HI":
                        if "say_hi_time" not in st.session_state:
                            st.session_state.say_hi_time = time.time()

                        elapsed = time.time() - st.session_state.say_hi_time
                        countdown = max(0, int(3 - elapsed))

                        if countdown > 0:
                            countdown_placeholder.markdown(f"""
                                <h2 style='text-align:center; color:#FF6347'>⏳ {countdown} giây nữa chụp ảnh!</h2>
                            """, unsafe_allow_html=True)
                        else:
                            st.image(debug_image, caption="📸 Ảnh chụp đã chụp!", channels="BGR", use_container_width=True)
                            if "captured_images" not in st.session_state:
                                st.session_state.captured_images = []
                            st.session_state.captured_images.append(debug_image.copy())
                            st.session_state.show_save_button = True
                            countdown_placeholder.empty()
                            del st.session_state.say_hi_time
                    else:
                        if "say_hi_time" in st.session_state:
                            del st.session_state.say_hi_time
                        countdown_placeholder.empty()

            debug_image = draw_info(debug_image, fps)
            img_containter.image(debug_image, channels="BGR")
            cv.waitKey(10)

        cap.release()
        cv.destroyAllWindows()

    if st.session_state.get("show_save_button") and "captured_images" in st.session_state:
        if st.button("💾 Lưu tất cả ảnh", key="save_hi_image_button"):
            save_dir = os.path.join("assets", "captures")
            os.makedirs(save_dir, exist_ok=True)

            for idx, img in enumerate(st.session_state.captured_images):
                filename = f"say_hi_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.jpg"
                save_path = os.path.join(save_dir, filename)
                cv.imwrite(save_path, img)

            st.success(f"✅ Đã lưu {len(st.session_state.captured_images)} ảnh vào thư mục: {save_dir}")

            del st.session_state.captured_images
            del st.session_state.show_save_button


app()

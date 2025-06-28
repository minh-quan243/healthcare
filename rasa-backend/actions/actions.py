from actions.xai_icfts import icfts_explanation
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import joblib
import pandas as pd
import logging
import os
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
# from xai_icfts import icfts_explanation

# Thiết lập logging
logger = logging.getLogger(__name__)
print("Imported icfts_explanation:", icfts_explanation)
class ActionPredictHeartDisease(Action):
    def name(self) -> Text:
        return "action_predict_heart_disease"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            # Load trained model - kiểm tra file tồn tại trước
            model_path = r'D:\Pycharm\chat\rasa-backend\models\heart_disease_model_logistic.pkl'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            model = joblib.load(model_path)

            # Lấy thông tin từ slots
            slot_mapping = {
                'age': ("age", float),
                'sex': ("sex", int),
                'cp': ("chest_pain_type", int),
                'trestbps': ("resting_bp", float),
                'chol': ("cholesterol", float),
                'fbs': ("fasting_blood_sugar", int),
                'restecg': ("resting_ecg", int),
                'thalach': ("max_heart_rate", float),
                'exang': ("exercise_angina", int),
                'oldpeak': ("st_depression", float),
                'slope': ("st_slope", int),
                'ca': ("num_major_vessels", int),
                'thal': ("thalassemia", int)
            }

            slots = {}
            missing = []

            for feature, (slot_name, dtype) in slot_mapping.items():
                slot_value = tracker.get_slot(slot_name)
                if slot_value is None:
                    missing.append(slot_name)
                    continue

                try:
                    # Chuyển đổi kiểu dữ liệu
                    slots[feature] = dtype(slot_value)
                except (ValueError, TypeError) as e:
                    logger.error(f"Error converting slot {slot_name} to {dtype}: {e}")
                    missing.append(slot_name)

            # Kiểm tra thiếu thông tin
            if missing:
                dispatcher.utter_message(
                    text=f"Thiếu hoặc sai định dạng thông tin về: {', '.join(missing)}. Vui lòng cung cấp đầy đủ và chính xác."
                )
                return []

            # Chuẩn bị dữ liệu đầu vào
            input_data = pd.DataFrame([slots])

            # Áp dụng one-hot encoding cho các biến phân loại
            categorical_cols = ['cp', 'restecg', 'slope', 'thal']
            input_data = pd.get_dummies(input_data, columns=categorical_cols)

            # Kiểm tra và thêm các cột còn thiếu
            try:
                expected_cols = model.feature_names_in_  # Thay vì model.named_steps[...]
                for col in expected_cols:
                    if col not in input_data.columns:
                        input_data[col] = 0

                # Sắp xếp đúng thứ tự
                input_data = input_data[expected_cols]
            except AttributeError as e:
                logger.error(f"Error processing model features: {e}")
                dispatcher.utter_message(
                    text="Có lỗi xảy ra khi xử lý dữ liệu đầu vào. Vui lòng thử lại sau."
                )
                return []

            # Dự đoán
            try:
                probability = model.predict_proba(input_data)[0][1]
                risk_score = int(round(probability * 100))  # Làm tròn và chuyển thành phần trăm

                # Phân loại nguy cơ với ngưỡng tùy chỉnh
                risk_level, recommendation = self._assess_risk(probability)

                # Phản hồi chi tiết
                message = (
                    f"Kết quả đánh giá nguy cơ bệnh tim:\n\n"
                    f"📊 Xác suất: {probability * 100:.1f}%\n"
                    f"⚠️ Mức độ nguy cơ: {risk_level}\n"
                    f"💡 Khuyến nghị: {recommendation}\n\n"
                    f"Lưu ý: Đây chỉ là dự đoán tự động, không thay thế chẩn đoán của bác sĩ."
                )
                # --- Phần giải thích ICFTS ---
                dispatcher.utter_message(text=message)
                feature_names = model.feature_names_in_.tolist()
                explanation = icfts_explanation(model, feature_names, top_n=3)
                dispatcher.utter_message(text=explanation)
                # -----------------------------
                # ====== LIME explanation ======

                # Tạo explainer (giả định model là logistic regression, input là số liệu dạng numpy)
                explainer = LimeTabularExplainer(
                    training_data=np.array(input_data),
                    # Dùng chính input_data làm training_data cho đơn giản, thực tế nên dùng data train gốc
                    feature_names=expected_cols,
                    class_names=["Không bị bệnh", "Bệnh tim"],
                    discretize_continuous=True
                )

                exp = explainer.explain_instance(
                    data_row=input_data.iloc[0].values,
                    predict_fn=model.predict_proba,
                    num_features=3
                )

                # Lấy giải thích dạng text
                lime_exp = "💡 LIME giải thích các đặc trưng ảnh hưởng:\n"
                for feature, weight in exp.as_list():
                    direction = "tăng" if weight > 0 else "giảm"
                    lime_exp += f"- {feature}: trọng số {weight:.3f} → có xu hướng {direction} nguy cơ\n"

                dispatcher.utter_message(text=lime_exp)

                # ===============================

                # Cập nhật risk_score vào slot
                return [SlotSet("risk_score", risk_score)]

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                dispatcher.utter_message(
                    text="Có lỗi xảy ra khi dự đoán. Vui lòng kiểm tra lại thông tin hoặc thử lại sau."
                )
                return []

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            dispatcher.utter_message(
                text="Xin lỗi, có lỗi hệ thống xảy ra. Vui lòng thử lại sau."
            )
            return []

    def _assess_risk(self, probability: float) -> tuple:
        """Phân loại nguy cơ và đưa ra khuyến nghị"""
        if probability >= 0.75:
            return (
                "rất cao",
                "Bạn có nguy cơ rất cao mắc bệnh tim. Nên đi khám bác sĩ chuyên khoa tim mạch ngay."
            )
        elif probability >= 0.6:
            return (
                "cao",
                "Bạn có nguy cơ cao. Nên đặt lịch khám bác sĩ sớm và làm các xét nghiệm chuyên sâu."
            )
        elif probability >= 0.4:
            return (
                "trung bình",
                "Bạn có nguy cơ ở mức trung bình. Nên đi khám tổng quát và thay đổi lối sống."
            )
        elif probability >= 0.2:
            return (
                "thấp",
                "Nguy cơ thấp nhưng vẫn cần theo dõi. Duy trì lối sống lành mạnh và kiểm tra định kỳ."
            )
        else:
            return (
                "rất thấp",
                "Nguy cơ rất thấp. Tiếp tục duy trì lối sống khoa học để bảo vệ sức khỏe tim mạch."
            )
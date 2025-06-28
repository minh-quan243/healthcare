from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd

class LimeTabularExplainerWrapper:
    def __init__(self, model, X_train, feature_names, class_names=None):
        """
        Khởi tạo LimeTabularExplainer.

        Args:
            model: mô hình sklearn đã huấn luyện, có phương thức predict_proba
            X_train: dữ liệu train dạng numpy array hoặc pandas DataFrame
            feature_names: list tên các đặc trưng
            class_names: list tên các lớp (ví dụ ["Không bệnh", "Bệnh tim"])
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names if class_names else ["Class 0", "Class 1"]

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values

        self.explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=self.class_names,
            discretize_continuous=True
        )

    def explain(self, sample, num_features=5):
        """
        Giải thích một mẫu input.

        Args:
            sample: 1D numpy array hoặc pandas Series đại diện cho mẫu cần giải thích
            num_features: số đặc trưng muốn giải thích

        Returns:
            list of tuples: [(feature_name, weight), ...]
        """
        if hasattr(sample, "values"):
            sample = sample.values

        exp = self.explainer.explain_instance(
            data_row=sample,
            predict_fn=self.model.predict_proba,
            num_features=num_features
        )

        return exp.as_list()

    def explain_text(self, sample, num_features=5):
        """
        Trả về giải thích dạng text.

        Args:
            sample: 1D numpy array hoặc pandas Series đại diện cho mẫu cần giải thích
            num_features: số đặc trưng muốn giải thích

        Returns:
            str: giải thích dạng text
        """
        if hasattr(sample, "values"):
            sample = sample.values

        exp = self.explainer.explain_instance(
            data_row=sample,
            predict_fn=self.model.predict_proba,
            num_features=num_features
        )
        return exp.as_text()

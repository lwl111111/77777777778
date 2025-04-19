import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap  # 导入SHAP库
import matplotlib.pyplot as plt
from PIL import Image  # Ensure PIL.Image is imported properly

# 加载预训练的XGBoost模型
model = joblib.load('vote排名前6.pkl')

# 更新后的特征范围定义
feature_ranges = {
    "Sex": {"type": "categorical", "options": [0, 1]},
    'Long-standing illness or disability': {"type": "categorical", "options": [0, 1]},
    "Age": {"type": "numerical"},
    'Number of non-cancer illnesses': {"type": "numerical"},
    'Number of medications taken': {"type": "numerical"},
    "Systolic Blood Pressure": {"type": "numerical"},
    'Cholesterol ratio': {"type": "numerical"},
    "Plasma GDF15": {"type": "numerical"},
    "Plasma MMP12": {"type": "numerical"},
    "Plasma NTproBNP": {"type": "numerical"},
    "Plasma AGER": {"type": "numerical"},
    "Plasma PRSS8": {"type": "numerical"},
    "Plasma PSPN": {"type": "numerical"},
    "Plasma WFDC2": {"type": "numerical"},
    "Plasma LPA": {"type": "numerical"},
    "Plasma CXCL17": {"type": "numerical"},
    "Plasma GAST": {"type": "numerical"},
    "Plasma RGMA": {"type": "numerical"},
    "Plasma EPHA4": {"type": "numerical"},
}

# Streamlit界面标题
st.title("10-Year MACE Risk Prediction")

# 创建两个列，显示输入项
col1, col2 = st.columns(2)

feature_values = []

# 通过 feature_ranges 保持顺序
for i, (feature, properties) in enumerate(feature_ranges.items()):
    if properties["type"] == "numerical":
        # 数值型输入框
        if i % 2 == 0:
            with col1:
                value = st.number_input(
                    label=f"{feature}",
                    value=0.0,  # 默认值为0
                    key=f"{feature}_input"
                )
        else:
            with col2:
                value = st.number_input(
                    label=f"{feature}",
                    value=0.0,  # 默认值
                    key=f"{feature}_input"
                )
    elif properties["type"] == "categorical":
        if feature == "Sex":
            with col1:  # 将"Sex"放在第一个列中
                value = st.radio(
                    label="Sex",
                    options=[0, 1],  # 0 = Female, 1 = Male
                    format_func=lambda x: "Female" if x == 0 else "Male",
                    key=f"{feature}_input"
                )
        elif feature == 'Long-standing illness or disability':
            with col2:  # 将"Long-standing illness or disability"放在第二个列中
                value = st.radio(
                    label="Long-standing illness or disability",
                    options=[0, 1],  # 0 = No, 1 = Yes
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    key=f"{feature}_input"
                )
    feature_values.append(value)

# 用这个 feature_values 来生成特征数据框
features = np.array([feature_values])

# 创建特征数据框
feature_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())

# 初始化 SHAP 解释器
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(feature_df)

# 打印 SHAP 值的形状，确保是二维的
print(f"SHAP values shape: {np.shape(shap_values)}")

# 对于二分类问题，shap_values[0] 和 shap_values[1] 分别对应负类和正类的 SHAP 值
if isinstance(shap_values, list):
    # 二分类模型，shap_values[1] 是正类（MACE）的 SHAP 值
    shap_values_class_1 = shap_values[1]  # 关注正类（MACE）的 SHAP 值
    expected_value_class_1 = explainer.expected_value[1]  # 对应正类的期望值
else:
    # 如果只有一个输出，那么默认就使用正类的 SHAP 值
    shap_values_class_1 = shap_values[0]  # 对于二分类的单一输出
    expected_value_class_1 = explainer.expected_value

# 检查 shao_values_class_1 的维度
print(f"shap_values_class_1 shape: {np.shape(shap_values_class_1)}")

# 计算特征重要性并提取前 5 个重要特征
feature_importance = np.abs(shap_values_class_1).mean(0)
feature_indices = np.argsort(feature_importance)[::-1]  # 按照特征重要性排序

# 选择前 5 个重要特征
top_5_indices = feature_indices[:5]
top_5_features = feature_df.columns[top_5_indices]

# 计算剩余特征的 SHAP 值之和
if shap_values_class_1.ndim == 1:  # 如果 SHAP 值是 1D 数组
    other_shap_sum = np.sum(shap_values_class_1[feature_indices[5:]])  # 总和
else:
    other_shap_sum = np.sum(shap_values_class_1[:, feature_indices[5:]], axis=1)

# 创建一个新的数据框，仅包含前 5 个最重要的特征
selected_feature_df = feature_df[top_5_features]

# 创建 "Other" 标签列，不赋值给 "Other"
other_feature_df = pd.DataFrame({"Other": ["Other"] * len(feature_df)})

# 创建新的 SHAP 值数组，包含前 5 个特征的 SHAP 值和剩余特征的 SHAP 值之和
selected_shap_values = np.concatenate(
    [shap_values_class_1[top_5_indices], other_shap_sum.reshape(-1, 1)],
    axis=1
)

# 合并前 5 个特征和 "Other" 标签列
final_feature_df = pd.concat([selected_feature_df, other_feature_df], axis=1)

# 生成 SHAP 力图
shap.force_plot(
    expected_value_class_1,
    selected_shap_values,
    final_feature_df,
    matplotlib=True,
    show=False  # 防止图形立即显示
)

# 保存图像文件
image_path = "shap_force_plot_class_1_selected.png"

# 确保图像保存正确
plt.savefig(image_path, bbox_inches='tight', dpi=1200)

# 检查图像文件是否存在
import os
if os.path.exists(image_path):
    st.write(f"Image saved successfully at {image_path}")
else:
    st.error(f"Error: Image was not saved at {image_path}")

# 尝试打开并验证图像
try:
    img = Image.open(image_path)  # 确保使用 PIL 打开图像
    img.verify()  # 验证图像是否有效
    st.image(image_path)  # 在 Streamlit 中显示图像
except (IOError, SyntaxError) as e:
    st.error(f"Error opening image: {e}")
    print(f"Error opening image: {e}")
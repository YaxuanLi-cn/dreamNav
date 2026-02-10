import json
import numpy as np


def angle_diff(pred, true):
    diff = pred - true
    # 将差值归一化到 [-180, 180] 范围
    diff = (diff + 180) % 360 - 180
    return diff


def calculate_metrics(pred, true, is_angle=False):
    """计算 MSE 和 MAE"""
    pred = np.array(pred)
    true = np.array(true)
    if is_angle:
        diff = angle_diff(pred, true)
    else:
        diff = pred - true
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    return mse, mae


def main():
    with open('step1_seen.json', 'r') as f:
        data = json.load(f)
    
    pred_deg = data['pred_deg_num']
    true_deg = data['true_deg_num']
    pred_rag = data['pred_rag_num']
    true_rag = data['true_rag_num']
    
    # 计算 degree 的 MSE 和 MAE（角度需要考虑环形特性）
    deg_mse, deg_mae = calculate_metrics(pred_deg, true_deg, is_angle=True)
    print(f"Degree MSE: {deg_mse:.4f}")
    print(f"Degree MAE: {deg_mae:.4f}")
    
    # 计算 rag 的 MSE 和 MAE
    rag_mse, rag_mae = calculate_metrics(pred_rag, true_rag)
    print(f"Rag MSE: {rag_mse:.4f}")
    print(f"Rag MAE: {rag_mae:.4f}")


if __name__ == '__main__':
    main()

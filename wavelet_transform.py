import pywt
import torch
import torch.nn.functional as F
import numpy as np


def multi_scale(patch_feature):
    b, h, c = patch_feature.shape
    high_res_features = F.interpolate(patch_feature.transpose(1, 2), scale_factor=1.5, mode='linear', align_corners=True)
    high_res_features = high_res_features[:, :, :h].transpose(1, 2)
    low_res_features = F.adaptive_avg_pool1d(patch_feature.transpose(1, 2), output_size=384).transpose(1, 2)
    low_res_features_2 = F.adaptive_max_pool1d(patch_feature.transpose(1, 2), output_size=384).transpose(1, 2)
    fus = torch.cat([low_res_features, low_res_features_2], dim=2)
    res_features_fus = F.adaptive_max_pool1d(fus, output_size=768)
    attention = low_res_features + low_res_features_2
    attention_weight = F.softmax(torch.mean(attention, dim=1, keepdim=True), dim=2)
    fus_attention = attention_weight * low_res_features + (1 - attention_weight) * low_res_features_2
    # ***********************n************
    attention = fus_attention + res_features_fus
    attention = torch.sigmoid(attention)
    low_res_features_upsampled = F.interpolate(attention.transpose(1, 2), size=h, mode='linear', align_corners=True)
    low_res_features_upsampled = low_res_features_upsampled.transpose(1, 2)
    alpha = 0.01
    beta = 0.99
    fused_features_sum = low_res_features_upsampled * beta + high_res_features * alpha
    return fused_features_sum


def apply_wavelet_transform(tensor):
    wavelets = ['haar', 'sym2']
    batch_size, height, width = tensor.shape
    wavelet_transformed_features = []
    for wavelet in wavelets:
        wavelet_obj = pywt.Wavelet(wavelet)
        transformed_features = []
        for img in tensor:
            img_np = img.cpu().numpy()  # 转为 NumPy
            coeffs = pywt.swt2(img_np, wavelet_obj, level=1)  # 一级分解
            approx_level1 = coeffs[0][0]  # 第一级低频
            horizonal, vertical, diagonal = coeffs[0][1]  # 高频分量
            combined_feature = (
                approx_level1 * 0.1 +  # 第一级低频，整体结构
                (horizonal + vertical + diagonal) * 0.9  # 高频分量，捕捉异常边缘
            )
            transformed_features.append(
                torch.tensor(combined_feature, dtype=tensor.dtype, device=tensor.device)
            )
        wavelet_transformed_features.append(torch.stack(transformed_features))
    target_size = wavelet_transformed_features[0].shape[1:]
    adjusted_features = []
    for features in wavelet_transformed_features:
        resized_features = F.interpolate(features.unsqueeze(1), size=target_size, mode='bilinear', align_corners=False).squeeze(1)
        adjusted_features.append(resized_features)
    fused_features = sum(adjusted_features) / len(adjusted_features)
    wavelet = 'haar'
    batch_size, height, width = tensor.shape
    transformed_low_features = []
    wavelet1 = pywt.Wavelet(wavelet)
    for i in range(batch_size):
        img = tensor[i].cpu().numpy()
        coeffs1 = pywt.dwt2(img, wavelet1)
        cA1, (cH1, cV1, cD1) = coeffs1
        High = np.stack([cH1, cV1, cD1], axis=-1)
        High1 = np.mean(High, axis=-1)
        low_frequency = cA1 * 0.9 + High1 * 0.1
        transformed_low_features.append(low_frequency)
    transformed_features_low = torch.tensor(np.array(transformed_low_features), dtype=tensor.dtype, device=tensor.device)
    transformed_features_low_1 = F.interpolate(transformed_features_low.unsqueeze(1), size=(height, width), mode='bilinear', align_corners=False)
    transformed_features_low_1 = transformed_features_low_1.squeeze(1)
    output_features = transformed_features_low_1 * 0.99 + fused_features * 0.01
    output_features = multi_scale(output_features) * 0.1 + output_features * 0.9
    return output_features
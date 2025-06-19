# Stable Diffusion Performance Analysis

This repository contains performance measurements for Stable Diffusion model inference on an NVIDIA GeForce RTX 4060 GPU.

## System Specifications
```
NVIDIA GeForce RTX 4060
VRAM: 8GB GDDR6
Driver Version: 576.52
CUDA Version: 12.9
```

## Performance Benchmarks
Below are the performance measurements for various optimization approaches:

### Initial run:

| Name                    | Self CPU % | Self CPU  | CPU total % | CPU total | CPU time avg | Self CUDA | Self CUDA % | CUDA total | CUDA time avg | # of Calls |
| ----------------------- | ---------- | --------- | ----------- | --------- | ------------ | --------- | ----------- | ---------- | ------------- | ---------- |
| model_inference         | 23.54%     | 109.568s  | 100.00%     | 465.392s  | 465.392s     | 1.030s    | 0.22%       | 465.378s   | 465.378s      | 1          |
| aten::conv2d            | 4.20%      | 19.550s   | 10.71%      | 49.860s   | 10.101ms     | 61.547ms  | 0.01%       | 141.022s   | 28.570ms      | 4936       |
| aten::convolution       | 1.04%      | 4.833s    | 6.51%       | 30.310s   | 6.141ms      | 41.352ms  | 0.01%       | 140.961s   | 28.558ms      | 4936       |
| aten::\_convolution     | 1.65%      | 7.659s    | 5.47%       | 25.477s   | 5.162ms      | 62.020ms  | 0.01%       | 140.919s   | 28.549ms      | 4936       |
| aten::cudnn_convolution | 2.80%      | 13.024s   | 2.82%       | 13.101s   | 2.654ms      | 140.309s  | 30.15%      | 140.390s   | 28.442ms      | 4936       |
| aten::matmul            | 8.11%      | 37.726s   | 23.77%      | 110.646s  | 17.152ms     | 227.631ms | 0.05%       | 117.255s   | 18.176ms      | 6451       |
| aten::bmm               | 1.86%      | 8.644s    | 1.86%       | 8.644s    | 2.660ms      | 111.117s  | 23.88%      | 111.117s   | 34.190ms      | 3250       |
| aten::softmax           | 0.01%      | 46.091ms  | 0.04%       | 183.158ms | 112.713us    | 23.285ms  | 0.01%       | 107.203s   | 65.971ms      | 1625       |
| aten::\_softmax         | 0.03%      | 137.067ms | 0.03%       | 137.067ms | 84.349us     | 107.180s  | 23.03%      | 107.180s   | 65.957ms      | 1625       |
| aten::div\_             | 0.02%      | 110.778ms | 0.02%       | 110.778ms | 68.129us     | 62.728s   | 13.48%      | 62.728s    | 38.578ms      | 1626       |

**Self CPU time total: 465.392s**  
**Self CUDA time total: 465.378s**

### Second Run (with torch.compile):

| Name                                               | Self CPU % | Self CPU    | CPU total % | CPU total  | CPU time avg | Self CUDA  | Self CUDA % | CUDA total | CUDA time avg | # of Calls |
| -------------------------------------------------- | ---------- | ----------- | ----------- | ---------- | ------------ | ---------- | ----------- | ---------- | ------------- | ---------- |
| model_inference                                    | 0.00%      | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 204.634s   | 105.62%     | 204.634s   | 102.317s      | 2          |
| model_inference                                    | 0.08%      | 169.650ms   | 100.00%     | 204.646s   | 204.646s     | 0.000us    | 0.00%       | 194.405s   | 194.405s      | 1          |
| Torch-Compiled Region: 1/0                         | 0.52%      | 1.068s      | 4.20%       | 8.588s     | 171.752ms    | 739.421ms  | 0.38%       | 185.616s   | 3.712s        | 50         |
| aten::bmm                                          | 0.07%      | 134.943ms   | 0.10%       | 196.386ms  | 58.448us     | 87.831s    | 45.33%      | 88.376s    | 26.303ms      | 3360       |
| triton_red_fused__softmax_14                       | 0.00%      | 3.053ms     | 0.00%       | 7.021ms    | 28.084us     | 48.017s    | 24.78%      | 48.017s    | 192.069ms     | 250        |
| triton_red_fused__softmax_14                       | 0.00%      | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 48.017s    | 24.78%      | 48.017s    | 192.069ms     | 250        |
| void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_... | 0.00% | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 46.017s    | 23.75%      | 46.017s    | 76.695ms      | 600        |
| void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_... | 0.00% | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 41.475s    | 21.41%      | 41.475s    | 41.475ms      | 1000       |
| aten::mm                                           | 0.14%      | 284.051ms   | 0.23%       | 474.624ms  | 60.935us     | 30.687s    | 15.84%      | 30.697s    | 3.941ms       | 7789       |
| aten::convolution                                  | 0.06%      | 126.019ms   | 2.22%       | 4.535s     | 885.061us    | 0.000us    | 0.00%       | 15.304s    | 2.987ms       | 5124       |

**Self CPU time total: 204.646s**  
**Self CUDA time total: 193.743s**

### Third Run (using pytorch FlashAttention)

> For some reason it performs worst, I have to look into it

| Name | Self CPU % | Self CPU | CPU total % | CPU total | CPU time avg | Self CUDA | Self CUDA % | CUDA total | CUDA time avg | # of Calls |
| ---------------------------------------------- | ---------- | ----------- | ----------- | ---------- | ------------ | ---------- | ----------- | ---------- | ------------- | ---------- |
| model_inference | 0.00% | 0.000us | 0.00% | 0.000us | 0.000us | 261.912s | 103.94% | 261.912s | 130.956s | 2 |
| model_inference | 0.06% | 153.544ms | 100.00% | 261.921s | 261.921s | 0.000us | 0.00% | 252.127s | 252.127s | 1 |
| Torch-Compiled Region: 1/0 | 0.33% | 860.456ms | 3.43% | 8.993s | 179.853ms | 674.732ms | 0.27% | 243.138s | 4.863s | 50 |
| aten::_scaled_dot_product_efficient_attention | 0.02% | 56.294ms | 0.08% | 208.764ms | 124.264us | 0.000us | 0.00% | 204.059s | 121.464ms | 1680 |
| aten::_efficient_attention_forward | 0.01% | 37.197ms | 0.04% | 112.254ms | 69.079us | 204.056s | 80.98% | 204.059s | 125.575ms | 1625 |
| fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEf... | 0.00% | 0.000us | 0.00% | 0.000us | 0.000us | 203.299s | 80.68% | 203.299s | 387.975ms | 524 |
| aten::mm | 0.11% | 282.933ms | 0.18% | 469.762ms | 60.311us | 22.373s | 8.88% | 22.379s | 2.873ms | 7789 |
| aten::convolution | 0.05% | 119.894ms | 2.34% | 6.122s | 1.195ms | 0.000us | 0.00% | 11.614s | 2.267ms | 5124 |
| aten::_convolution | 0.02% | 46.975ms | 2.28% | 5.978s | 1.211ms | 0.000us | 0.00% | 11.614s | 2.353ms | 4936 |
| aten::cudnn_convolution | 0.25% | 649.959ms | 2.26% | 5.931s | 1.202ms | 11.478s | 4.56% | 11.614s | 2.353ms | 4936 |

**Self CPU time total: 261.921s**  
**Self CUDA time total: 251.978s**

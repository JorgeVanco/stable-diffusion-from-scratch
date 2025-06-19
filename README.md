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

> For some reason it performs worst, I have to look into it. Maybe because it is using float32.

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

### Fourth Run (using torch.inference_mode instead of torch.no_grad)

| Name                                               | Self CPU % | Self CPU    | CPU total % | CPU total  | CPU time avg | Self CUDA  | Self CUDA % | CUDA total | CUDA time avg | # of Calls |
| -------------------------------------------------- | ---------- | ----------- | ----------- | ---------- | ------------ | ---------- | ----------- | ---------- | ------------- | ---------- |
| model_inference                                    | 0.00%      | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 202.237s   | 106.04%     | 202.237s   | 101.119s      | 2          |
| model_inference                                    | 0.08%      | 164.374ms   | 100.00%     | 202.247s   | 202.247s     | 0.000us    | 0.00%       | 191.259s   | 191.259s      | 1          |
| Torch-Compiled Region: 1/0                         | 0.52%      | 1.060s      | 4.38%       | 8.853s     | 177.065ms    | 580.683ms  | 0.30%       | 183.095s   | 3.662s        | 50         |
| aten::bmm                                          | 0.06%      | 115.354ms   | 0.09%       | 178.184ms  | 54.110us     | 87.487s    | 45.87%      | 87.975s    | 26.716ms      | 3293       |
| triton_red_fused__softmax_14                       | 0.00%      | 3.362ms     | 0.00%       | 8.194ms    | 32.774us     | 47.738s    | 25.03%      | 47.738s    | 190.952ms     | 250        |
| triton_red_fused__softmax_14                       | 0.00%      | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 47.738s    | 25.03%      | 47.738s    | 190.952ms     | 250        |
| void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_... | 0.00% | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 45.985s    | 24.11%      | 45.985s    | 76.642ms      | 600        |
| void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_... | 0.00% | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 41.272s    | 21.64%      | 41.272s    | 41.272ms      | 1000       |
| aten::mm                                           | 0.13%      | 262.772ms   | 0.23%       | 461.871ms  | 59.727us     | 30.027s    | 15.74%      | 30.031s    | 3.884ms       | 7733       |
| aten::convolution                                  | 0.04%      | 78.898ms    | 2.47%       | 4.986s     | 988.545us    | 0.000us    | 0.00%       | 14.596s    | 2.894ms       | 5044       |

**Self CPU time total: 202.247s**  
**Self CUDA time total: 190.709s**

### Fifth Run (using bfloat16 precision instead of float32)

| Name                                               | Self CPU % | Self CPU    | CPU total % | CPU total  | CPU time avg | Self CUDA  | Self CUDA % | CUDA total | CUDA time avg | # of Calls |
| -------------------------------------------------- | ---------- | ----------- | ----------- | ---------- | ------------ | ---------- | ----------- | ---------- | ------------- | ---------- |
| model_inference                                    | 0.00%      | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 91.781s    | 113.64%     | 91.781s    | 45.890s       | 2          |
| model_inference                                    | 0.15%      | 140.575ms   | 100.00%     | 91.789s    | 91.789s      | 0.000us    | 0.00%       | 81.191s    | 81.191s       | 1          |
| Torch-Compiled Region: 1/0                         | 1.18%      | 1.079s      | 6.73%       | 6.179s     | 123.583ms    | 594.311ms  | 0.74%       | 79.900s    | 1.598s        | 50         |
| aten::bmm                                          | 1.01%      | 928.399ms   | 1.08%       | 992.944ms  | 301.532us    | 42.739s    | 52.92%      | 43.059s    | 13.076ms      | 3293       |
| triton_red_fused__softmax_14                       | 0.00%      | 3.608ms     | 0.01%       | 8.074ms    | 32.296us     | 23.096s    | 28.60%      | 23.096s    | 92.386ms      | 250        |
| triton_red_fused__softmax_14                       | 0.00%      | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 23.096s    | 28.60%      | 23.096s    | 92.386ms      | 250        |
| void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681... | 0.00% | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 22.576s    | 27.95%      | 22.576s    | 45.151ms      | 500        |
| ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_3... | 0.00% | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 20.101s    | 24.89%      | 20.101s    | 20.081ms      | 1001       |
| aten::mm                                           | 0.29%      | 263.086ms   | 0.48%       | 439.652ms  | 56.825us     | 6.742s     | 8.35%       | 6.743s     | 871.483us     | 7737       |
| aten::convolution                                  | 0.24%      | 224.165ms   | 1.45%       | 1.327s     | 263.000us    | 0.000us    | 0.00%       | 3.125s     | 619.592us     | 5044       |

**Self CPU time total: 91.789s**  
**Self CUDA time total: 80.765s**

### Sixth Run (using float16 precision instead of float32)
| Name                                            | Self CPU % | Self CPU  | CPU total % | CPU total | CPU time avg | Self CUDA | Self CUDA % | CUDA total | CUDA time avg | # of Calls |
|-------------------------------------------------|------------|-----------|--------------|------------|----------------|------------|---------------|-------------|----------------|-------------|
| model_inference                                 | 0.00%      | 0.000us   | 0.00%        | 0.000us    | 0.000us        | 96.659s    | 113.27%       | 96.659s     | 48.330s         | 2           |
| model_inference                                 | 0.16%      | 159.394ms | 100.00%      | 96.672s    | 96.672s        | 0.000us    | 0.00%         | 85.829s     | 85.829s         | 1           |
| Torch-Compiled Region: 1/0                      | 1.28%      | 1.234s    | 7.02%        | 6.789s     | 135.773ms      | 505.302ms | 0.59%         | 84.343s     | 1.687s          | 50          |
| aten::bmm                                       | 1.05%      | 1.013s    | 1.12%        | 1.080s     | 327.956us      | 45.040s    | 52.78%        | 45.453s     | 13.803ms        | 3293        |
| void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816...| 0.00%      | 0.000us   | 0.00%        | 0.000us    | 0.000us        | 23.715s    | 27.79%        | 23.715s     | 94.859ms        | 250         |
| triton_red_fused__softmax_14                   | 0.00%      | 3.748ms   | 0.01%        | 8.529ms    | 34.116us       | 23.662s    | 27.73%        | 23.662s     | 94.647ms        | 250         |
| triton_red_fused__softmax_14                   | 0.00%      | 0.000us   | 0.00%        | 0.000us    | 0.000us        | 23.662s    | 27.73%        | 23.662s     | 94.647ms        | 250         |
| ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_nn  | 0.00%      | 0.000us   | 0.00%        | 0.000us    | 0.000us        | 21.127s    | 24.76%        | 21.127s     | 84.510ms        | 250         |
| aten::mm                                        | 0.29%      | 284.905ms | 0.52%        | 499.538ms  | 64.565us       | 6.585s     | 7.72%         | 6.585s      | 851.102us       | 7737        |
| aten::convolution                               | 0.08%      | 79.102ms  | 1.55%        | 1.495s     | 296.430us      | 0.000us    | 0.00%         | 6.406s      | 1.270ms         | 5044        |

**Self CPU time total: 96.672s**  
**Self CUDA time total: 85.332s**

### Sixth Run (Using pytorch FlashAttention with bloat16)
**Inference in under a minute!!!**

| Name                                                    | Self CPU % | Self CPU   | CPU total % | CPU total  | CPU time avg | Self CUDA  | Self CUDA % | CUDA total | CUDA time avg | # of Calls |
|---------------------------------------------------------|------------|------------|-------------|------------|---------------|------------|--------------|-------------|----------------|-------------|
| model_inference                                         | 0.00%      | 0.000us    | 0.00%       | 0.000us    | 0.000us       | 93.961s    | 270.83%      | 93.961s     | 46.980s        | 2           |
| model_inference                                         | 0.16%      | 146.259ms  | 100.00%     | 93.968s    | 93.968s       | 0.000us    | 0.00%        | 35.056s     | 35.056s        | 1           |
| Torch-Compiled Region: 1/0                              | 0.95%      | 891.482ms  | 11.65%      | 10.943s    | 218.856ms     | 3.115s     | 8.98%        | 20.925s     | 418.491ms      | 50          |
| InductorBenchmarker.benchmark_gpu (dynamo_timed)        | 0.00%      | 0.000us    | 0.00%       | 0.000us    | 0.000us       | 20.242s    | 58.35%       | 20.242s     | 55.764ms       | 363         |
| Torch-Compiled Region: 2/0                              | 0.03%      | 30.361ms   | 14.19%      | 13.331s    | 13.331s       | 9.849s     | 28.39%       | 12.840s     | 12.840s        | 1           |
| CachingAutotuner.benchmark_all_configs (dynamo_timed...)| 0.19%      | 174.454ms  | 22.95%      | 21.570s    | 209.413ms     | 0.000us    | 0.00%        | 6.794s      | 65.958ms       | 103         |
| InductorBenchmarker.benchmark_gpu (dynamo_timed)        | 2.74%      | 2.576s     | 22.77%      | 21.395s    | 58.940ms      | 0.000us    | 0.00%        | 6.794s      | 18.715ms       | 363         |
| aten::zero_                                             | 0.42%      | 398.503ms  | 1.60%       | 1.508s     | 21.568us      | 0.000us    | 0.00%        | 6.703s      | 95.887us       | 69905       |
| aten::fill_                                             | 0.36%      | 341.773ms  | 1.18%       | 1.109s     | 17.078us      | 6.699s     | 19.31%       | 6.703s      | 103.204us      | 64949       |
| void at::native::vectorized_elementwise_kernel<4, at... | 0.00%      | 0.000us    | 0.00%       | 0.000us    | 0.000us       | 6.699s     | 19.31%       | 6.699s      | 103.147us      | 64949       |

**Self CPU time total: 93.968s**  
**Self CUDA time total: 34.694s**

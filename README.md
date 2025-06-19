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

| Name                                                   | Self CPU % | Self CPU    | CPU total % | CPU total  | CPU time avg | Self CUDA   | Self CUDA % | CUDA total  | CUDA time avg | # of Calls |
| ------------------------------------------------------ | ---------- | ----------- | ----------- | ---------- | ------------ | ----------- | ----------- | ----------- | ------------- | ---------- |
| model_inference                                        | 0.00%      | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 256.105s    | 130.68%     | 256.105s    | 128.053s      | 2          |
| model_inference                                        | 0.05%      | 131.109ms   | 100.00%     | 256.110s   | 256.110s     | 0.000us     | 0.00%       | 200.049s    | 200.049s      | 1          |
| Torch-Compiled Region: 1/0                             | 0.41%      | 1.047s      | 5.43%       | 13.910s    | 278.200ms    | 3.581s      | 1.83%       | 181.936s    | 3.639s        | 50         |
| aten::bmm                                              | 0.16%      | 412.788ms   | 0.22%       | 552.780ms  | 103.459us    | 88.105s     | 44.96%      | 88.126s     | 16.494ms      | 5343       |
| triton_red_fused__softmax_14                           | 0.00%      | 3.062ms     | 0.00%       | 7.117ms    | 28.469us     | 47.684s     | 24.33%      | 47.684s     | 190.734ms     | 250        |
| triton_red_fused__softmax_14                           | 0.00%      | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 47.684s     | 24.33%      | 47.684s     | 190.734ms     | 250        |
| void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_... | 0.00%      | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 45.974s     | 23.46%      | 45.974s     | 38.569ms      | 1192       |
| void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_... | 0.00%      | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 41.612s     | 21.23%      | 41.612s     | 39.036ms      | 1066       |
| aten::mm                                               | 0.25%      | 632.716ms   | 0.37%       | 936.869ms  | 102.200us    | 23.480s     | 11.98%      | 23.495s     | 2.563ms       | 9167       |
| aten::convolution                                      | 0.25%      | 645.138ms   | 1.29%       | 3.297s     | 528.088us    | 0.000us     | 0.00%       | 14.923s     | 2.390ms       | 6244       |

**Self CPU time total: 256.110s**  
**Self CUDA time total: 195.978s**

### Third Run (using pytorch FlashAttention)

| Name                                                   | Self CPU % | Self CPU    | CPU total % | CPU total  | CPU time avg | Self CUDA   | Self CUDA % | CUDA total  | CUDA time avg | # of Calls |
| ------------------------------------------------------ | ---------- | ----------- | ----------- | ---------- | ------------ | ----------- | ----------- | ----------- | ------------- | ---------- |
| model_inference                                        | 0.00%      | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 169.153s    | 161.10%     | 169.153s    | 84.576s       | 2          |
| model_inference                                        | 0.08%      | 135.853ms   | 100.00%     | 169.166s   | 169.166s     | 0.000us     | 0.00%       | 105.270s    | 105.270s      | 1          |
| Torch-Compiled Region: 1/0                             | 0.61%      | 1.037s      | 8.40%       | 14.211s    | 284.212ms    | 3.952s      | 3.76%       | 54.988s     | 1.100s        | 50         |
| InductorBenchmarker.benchmark_gpu (dynamo_timed)       | 0.00%      | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 52.855s     | 50.34%      | 52.855s     | 120.126ms     | 440        |
| Torch-Compiled Region: 2/0                             | 0.02%      | 26.273ms    | 27.77%      | 46.973s    | 46.973s      | 37.211s     | 35.44%      | 46.642s     | 46.642s       | 1          |
| aten::mm                                               | 0.35%      | 600.125ms   | 0.53%       | 888.909ms  | 101.962us    | 28.612s     | 27.25%      | 28.616s     | 3.282ms       | 8718       |
| aten::convolution                                      | 0.37%      | 621.182ms   | 1.01%       | 1.712s     | 274.219us    | 0.000us     | 0.00%       | 13.935s     | 2.232ms       | 6244       |
| aten::_convolution                                     | 0.03%      | 43.885ms    | 0.53%       | 888.199ms  | 179.943us    | 0.000us     | 0.00%       | 13.935s     | 2.823ms       | 4936       |
| aten::cudnn_convolution                                | 0.31%      | 525.592ms   | 0.50%       | 844.314ms  | 171.052us    | 13.830s     | 13.17%      | 13.935s     | 2.823ms       | 4936       |
| triton_poi_fused_convolution_native_group_norm_silu_... | 0.00%      | 0.000us     | 0.00%       | 0.000us    | 0.000us      | 10.784s     | 10.27%      | 10.784s     | 250.800ms     | 43         |

**Self CPU time total: 169.166s**  
**Self CUDA time total: 104.996s**
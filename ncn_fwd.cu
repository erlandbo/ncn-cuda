#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


__device__ int extract_group_index_(int i, int l, int* strides, int n) {
    // strides [l] equals n ** l.
    //  lower part : digits 0 ... l -1
    int lower = i % strides[l];
    // the digit at position l
    int index_within_group = (i / strides[l]) % n;
    // digits above position l , shifted down one place
    int upper = i / (n * strides[l]);
    // reassemble the group number without i_l
    int group_number = lower + upper * strides[l];
    return index_within_group, group_number;
}

__device__ int extract_group_index_simple(int group_nr, int idx_within_group, int n, int N) {
    int i = group_nr * n + idx_within_group;
    return i;
}


__device__ float kernel(int group_nr, int idx_within_group, int n, int N) {
    int i = group_nr * n + idx_within_group;
    return i;
}


__global__
void forward_kernel(const float* X, const float* Xa, const float* W, const int B, const int N,
                    const int E, const int n, const int C, const int activation, const float alpha,
                    float *Y, float* Ya, const int l) {
    int tx = threadIdx.x;
    int bxy = blockIdx.x; int bn = blockIdx.y;  // batch, head index and group index
    int bx = bxy / C; int by = bxy % C;

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = n * E / C;  // size of x
    float* Xi = sram;
    float* Xia = &sram[tile_size];
    float* F = &sram[tile_size * 2];


    int i = extract_group_index_simple(bn, tx, n, N);
    int xi_offset = (bx * N * E) + (i * E) + by * (E / C);
    //if (bx == 0 and by == 0){
    //    printf("tx: %d  bx:%d by:%d bn:%d xi-offset:%d x0:%f x1:%f\n", tx, bx, by, bn, xi_offset, X[xi_offset + 0], X[xi_offset + 1]);
    //}


    for (int feat = 0; feat < E / C; feat++) {
        // Offset into x - different for each batch and head
        // Load to SRAM
        Xi[(tx * E / C) + feat] = X[xi_offset + feat];
        Xia[(tx * E / C) + feat] = Xa[xi_offset + feat];

        //if (bx == 0 and by == 0){
        //    printf("tx: %d  bx:%d by:%d bn:%d xi:%f x:%f \n", tx, bx, by, bn, Xi[tx * E / C + feat], X[xi_offset + feat]);
        //}

    }
    
    __syncthreads();  // such that the inner loop can use the correct Xi, Xia
    
    for (int j = 0; j < n; j++)  {
        
        float attn = 0.0;
        for (int feat = 0; feat < E / C; feat++) {
            attn += Xi[tx * E / C + feat] * W[E / C * by + feat] + Xi[j * E / C + feat] * W[E + E / C * by + feat]; 
        }
        for (int feat = 0; feat < E / C; feat++) {
            float t = alpha * Xi[tx * E / C + feat] + (1.0-alpha) * attn * Xi[j * E / C + feat];
            float f;    
            if (activation == 0){
                f = tanh(t);
            }else if (activation == 1){
                f = max(t, 0.0);
            }else if (activation == 2){
                f = 1.0 / (1.0 + exp(-t));
            }
            else {
                f = t;
            }

            F[tx * E / C + feat] = f; 

        }
        
        __syncthreads();
        for (int feat = 0; feat < E / C; feat++) {
            float m = 0.9;
            float ya = m * Xia[tx * E / C + feat] + (1.0-m) * F[tx * E / C + feat]; 
            float yi = m * Xi[tx * E / C + feat] + (1.0-m) * ya;
            Xia[tx * E / C + feat] = ya; 
            Xi[tx * E / C + feat] = yi; 
        }
        __syncthreads();
    }
    for (int feat = 0; feat < E / C; feat++) {
        Y[xi_offset + feat] = Xi[(tx * E / C) + feat];
        Ya[xi_offset + feat] = Xia[(tx * E / C) + feat];
    }
}


std::tuple< torch::Tensor, torch::Tensor > forward(
    torch::Tensor X, 
    torch::Tensor Xa, 
    torch::Tensor W, 
    const float alpha, 
    const int activation, 
    const int n, 
    const int C,
    const int l
) {

    const int B = X.size(0); const int N = X.size(1); const int E = X.size(2);

    auto Y = torch::zeros_like(X);
    auto Ya = torch::zeros_like(Xa);
    torch::Device device(torch::kCUDA);

    // Calculate SRAM size needed per block
    int tile_size = n * E / C;
    const int sram_size = (3 * tile_size * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B * C, N / n);  // batch_size x num_heads
    dim3 block_dim(n);  // Bc threads per block

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        X.data_ptr<float>(), Xa.data_ptr<float>(), W.data_ptr<float>(), 
        B, N, E, n, C, activation, alpha,
        Y.data_ptr<float>(), Ya.data_ptr<float>(), l
    );
    return std::make_tuple(Y, Ya);
}
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>



__device__ int extract_group_index_simple_fwd(int group_nr, int idx_within_group, int n, int N) {
    int i = group_nr * n + idx_within_group;
    return i;
}



__global__
void forward_kernel(
    const float* X, 
    const float* Xa, 
    const float* W, 
    const uint batch_dim, 
    const uint ctx_dim, 
    const uint embed_dim, 
    const uint cache_dim, 
    const uint activation, 
    const float alpha,
    const uint nhead,
    float *Y, 
    float* Ya, 
    const uint module_l
) {
    const uint tx = threadIdx.x;
    const uint bbatchhead = blockIdx.x;
    const uint bcache = blockIdx.y;

    const uint bbatch = bbatchhead / nhead; 
    const uint bhead = bbatchhead % nhead;

    const uint nfeats = embed_dim / nhead;

    // Define SRAM
    const uint tile_size = cache_dim * nfeats;

    extern __shared__ float sram[];
    float* X_smem = sram;
    float* Xa_smem = &sram[tile_size];

    const uint i = extract_group_index_simple_fwd(bcache, tx, cache_dim, ctx_dim);
    const uint xi_offset = (bbatch * ctx_dim * embed_dim) + (i * embed_dim) + bhead * nfeats;


    for (int feat = 0; feat < nfeats; feat++) {
        X_smem[tx * nfeats + feat] = X[xi_offset + feat];
        Xa_smem[tx * nfeats + feat] = Xa[xi_offset + feat];

    }
    
    __syncthreads();  // such that the inner loop can use the correct Xi, Xia
    
    float m = 0.9;
    for (int j = 0; j < nfeats; j++)  {
        __syncthreads();  // such that the inner loop can use the correct Xi, Xia

        float attn = 0.0;
        for (int feat = 0; feat < nfeats; feat++) {
            attn += X_smem[tx * nfeats + feat] * W[nfeats * bhead + feat] \
                + X_smem[j * nfeats + feat] * W[embed_dim + nfeats * bhead + feat]; 
        }
        __syncthreads();
        for (int feat = 0; feat < nfeats; feat++) {
            float t = alpha * X_smem[tx * nfeats + feat] + (1.0-alpha) * attn * X_smem[j * nfeats + feat];
            __syncthreads();
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

            float ya = m * Xa_smem[tx * nfeats + feat] + (1.0-m) * f; 
            float yi = m * X_smem[tx * nfeats + feat] + (1.0-m) * ya;
            Xa_smem[tx * nfeats + feat] = ya;
            X_smem[tx * nfeats + feat] = yi;
            __syncthreads();
        }
        
        __syncthreads();

    }
    __syncthreads();
    for (int feat = 0; feat < nfeats; feat++) {
        Y[xi_offset + feat] = X_smem[tx * nfeats + feat];
        Ya[xi_offset + feat] = Xa_smem[tx * nfeats + feat];
    }
}


std::vector< torch::Tensor > forward(
    torch::Tensor X, 
    torch::Tensor Xa, 
    torch::Tensor W, 
    const float alpha, 
    const int activation, 
    const int cache_dim, 
    const int nhead,
    const int module_l
) {

    const int batch_dim = X.size(0); const int ctx_dim = X.size(1); const int embed_dim = X.size(2);

    auto Y = torch::zeros_like(X);
    auto Ya = torch::zeros_like(Xa);
    torch::Device device(torch::kCUDA);

    // Calculate SRAM size needed per block
    int tile_size = cache_dim * embed_dim / nhead;
    const int sram_size = (2 * tile_size * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(batch_dim * nhead, ctx_dim / cache_dim);
    dim3 block_dim(cache_dim);  // Bc threads per block

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        X.data_ptr<float>(), 
        Xa.data_ptr<float>(), 
        W.data_ptr<float>(), 
        batch_dim, 
        ctx_dim, 
        embed_dim, 
        cache_dim, 
        activation, 
        alpha,
        nhead,
        Y.data_ptr<float>(), 
        Ya.data_ptr<float>(), 
        module_l
    );

    cudaDeviceSynchronize();

    return {Y, Ya};
}

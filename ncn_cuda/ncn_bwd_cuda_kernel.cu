#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ int extract_index_group_v2_bwd(int group_nr, int idx_within_group, int n, int N, int module_l) {
    const uint num_groups = N / n;
    uint stride;
    if (module_l == 0){
        stride = 0;
    }else{
        stride = int((1ULL << (module_l - 1)) % (uint64_t)num_groups);
    }

    const uint64_t chunk = (uint64_t(group_nr) + uint64_t(idx_within_group) * uint64_t(stride)) % uint64_t(num_groups);
    return int(chunk) * n + idx_within_group;
}


__device__ int extract_group_index_simple_bwd(int group_nr, int idx_within_group, int n, int N) {
    int i = group_nr * n + idx_within_group;
    return i;
}




__global__
void backward_kernel(
    const float* Y, 
    const float* Ya, 
    const float* W, 
    const float*dY, 
    const float*dYa,
    const uint batch_dim, 
    const uint ctx_dim, 
    const uint embed_dim, 
    const uint cache_dim, 
    const uint activation, 
    const float alpha,
    const uint nhead,
    float *X, 
    float* Xa, 
    float *dX, 
    float* dXa, 
    float* dW, 
    const uint module_l
) {
    const uint tx = threadIdx.x;
    const uint bbatchhead = blockIdx.x;
    const uint bcache = blockIdx.y;

    const uint bbatch = bbatchhead / nhead; 
    const uint bhead = bbatchhead % nhead;

    const uint nfeats = embed_dim / nhead;
    // Define SRAM
    extern __shared__ float sram[];
    const uint tile_size = cache_dim * nfeats;
    float* Y_smem = sram;
    float* Ya_smem = &sram[tile_size];
    float* dxj_smem = &sram[tile_size * 2];

    float* dY_smem = &sram[tile_size * 3];
    float* dYa_smem = &sram[tile_size * 4];
    float* dWi_smem = &sram[tile_size * 5];
    float* dWj_smem = &sram[tile_size * 6];

    const uint i = extract_index_group_v2_bwd(bcache, tx, cache_dim, ctx_dim, module_l);
    const uint yi_offset = (bbatch * ctx_dim * embed_dim) + (i * embed_dim) + bhead * nfeats;
    const uint dwi_offset = (bbatch * ctx_dim * 2*embed_dim) + (i * 2*embed_dim) + bhead * nfeats;
    const uint dwj_offset = (bbatch * ctx_dim * 2*embed_dim) + (i * 2*embed_dim) + bhead * nfeats + embed_dim;

    for (int feat = 0; feat < nfeats; feat++) {
        Y_smem[tx * nfeats + feat] = Y[yi_offset + feat];
        Ya_smem[tx * nfeats + feat] = Ya[yi_offset + feat];
        dY_smem[tx * nfeats + feat] = dY[yi_offset + feat];
        dYa_smem[tx * nfeats + feat] = dYa[yi_offset + feat];
        dWi_smem[tx * nfeats + feat] = 0.0f;
        dWj_smem[tx * nfeats + feat] = 0.0f;
    }
    
    __syncthreads();  // such that the inner loop can use the correct Y and Ya

    float m = 0.9;
    
    for (int j = cache_dim-1; j >= 0; j--)  {
        
        // Use reversibel trick xi
        for (int feat = 0; feat < nfeats; feat++) {
            Y_smem[tx * nfeats + feat] = (Y_smem[tx * nfeats + feat] - Ya_smem[tx * nfeats + feat] * (1.0 - m)) / m;
        }
        __syncthreads();
        
        float attn = 0.0;
        for (int feat = 0; feat < nfeats; feat++) {
            attn += Y_smem[tx * nfeats + feat] * W[nfeats * bhead + feat] \
                 + Y_smem[j * nfeats + feat] * W[embed_dim + nfeats * bhead + feat]; 
        }

        // Use reversibel trick xa
        for (int feat = 0; feat < nfeats; feat++) {
            float t = alpha * Y_smem[tx * nfeats + feat] + (1.0-alpha) * attn * Y_smem[j * nfeats + feat];
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
            Ya_smem[tx * nfeats + feat] = (Ya_smem[tx * nfeats + feat] - f * (1.0 - m)) / m;
        }

        __syncthreads();

        // Compute vjp 
        // dL/dxi = dL/dyi dyi/dxi + dL/dya dya/dxi 
        // dL/dxa = dL/dyi dyi/dxa + dL/dya dya/dxa 
        // dL/dxj = dL/dyi dyi/dxj + dL/dya dya/dxj 
        
        for (int col = 0; col < nfeats; col++) {
            float dxi = 0.0f;
            float dxa = 0.0f;
            float dxj = 0.0f;
            float dwi = 0.0f;
            float dwj = 0.0f;

            for (int row = 0; row < nfeats; row++) {
                float delta_ij = (tx == j) ? 1.0f : 0.0f;
                float delta_kl = (row == col) ? 1.0f : 0.0f;
                
                float t = alpha * Y_smem[tx * nfeats + row] + (1.0-alpha) * attn * Y_smem[j * nfeats + row];
                float f, df_dt;
                if (activation == 0){
                    f = tanh(t);
                    df_dt = 1.0 - f * f;
                }else if (activation == 1){
                    f = max(t, 0.0);
                    df_dt = (f > 0) ? 1.0 : 0.0;
                }else if (activation == 2){
                    f = 1.0 / (1.0 + exp(-t));
                    df_dt = f * (1.0 - f);
                }
                else {
                    f = t;
                    df_dt = 1.0;
                }

                // Compute dxi for correct row and col
                float dt_dxi = alpha * delta_kl + (1.0-alpha) * (\
                    (Y_smem[j * nfeats + row] * (W[nfeats * bhead + col] + delta_ij * W[embed_dim + nfeats * bhead + col])) + \
                    attn * delta_ij * delta_kl \
                );
                float df_dxi = df_dt * dt_dxi;
                
                float dya_dxi = (1.0-m) * df_dxi;
                float dyi_dxi = m * delta_kl + (1.0-m) * dya_dxi;
                dxi += dyi_dxi * dY_smem[tx * nfeats + row] + dya_dxi * dYa_smem[tx * nfeats + row];
                
                // Compute dxj when i!=j, contribution from other threads
                float dt_dxj = (1.0-alpha) * (\
                    Y_smem[j * nfeats + row] * (W[embed_dim + nfeats * bhead + col]) + \
                    attn * delta_kl \
                );
                float df_xj = df_dt * dt_dxj;
                float dya_dxj = (1.0-m) * df_xj;
                float dyi_dxj = (1.0-m) * dya_dxj;
                dxj += dyi_dxj * dY_smem[tx * nfeats + row] + dya_dxj * dYa_smem[tx * nfeats + row];

                // Compute dxa
                float dya_dxa = delta_kl * m;
                float dyi_dxa = (1.0-m) * dya_dxa;
                dxa += dyi_dxa * dY_smem[tx * nfeats + row] + dya_dxa * dYa_smem[tx * nfeats + row];

                // Compute dW
                float df_dwi = df_dt * (1.0-alpha) * Y_smem[tx * nfeats + col] * Y_smem[j * nfeats + row];  
                float df_dwj = df_dt * (1.0-alpha) * Y_smem[j * nfeats + col] * Y_smem[j * nfeats + row];  
                
                float dya_dwi = (1.0-m) * df_dwi;  
                float dya_dwj = (1.0-m) * df_dwj;  
                
                float dyi_dwi = (1.0-m) * dya_dwi;  
                float dyi_dwj = (1.0-m) * dya_dwj;
                
                dwi += dyi_dwi * dY_smem[tx * nfeats + row] + dya_dwi * dYa_smem[tx * nfeats + row];
                dwj += dyi_dwj * dY_smem[tx * nfeats + row] + dya_dwj * dYa_smem[tx * nfeats + row];

                __syncthreads();

            }

            __syncthreads();

            dWi_smem[tx * nfeats + col] += dwi;
            dWj_smem[tx * nfeats + col] += dwj;

            dxj_smem[tx * nfeats + col] = dxj;

            __syncthreads();

            for (int i = 0; i < cache_dim; i++) {
                if (tx == j){
                    if (i != tx){
                        dxi += dxj_smem[i * nfeats + col];
                    }
                }
            }
            __syncthreads();

            dY_smem[tx * nfeats + col] = dxi;
            dYa_smem[tx * nfeats + col] = dxa;
            __syncthreads();

        }
    }

    for (int feat = 0; feat < nfeats; feat++) {
        X[yi_offset + feat] = Y_smem[tx * nfeats + feat];
        Xa[yi_offset + feat] = Ya_smem[tx * nfeats + feat];
        dX[yi_offset + feat] = dY_smem[tx * nfeats + feat];
        dXa[yi_offset + feat] = dYa_smem[tx * nfeats + feat];
        dW[dwi_offset + feat] = dWi_smem[tx * nfeats + feat];
        dW[dwj_offset + feat] = dWj_smem[tx * nfeats + feat];
    }

}


std::vector< torch::Tensor > backward(
    torch::Tensor Y, 
    torch::Tensor Ya, 
    torch::Tensor dY, 
    torch::Tensor dYa, 
    torch::Tensor W, 
    const float alpha, 
    const int activation, 
    const int cache_dim, 
    const int nhead,
    const int module_l
) {

    const int batch_dim = Y.size(0); const int ctx_dim = Y.size(1); const int embed_dim = Y.size(2);

    auto X = torch::zeros_like(Y);
    auto Xa = torch::zeros_like(Ya);
    auto dX = torch::zeros_like(dY);
    auto dXa = torch::zeros_like(dYa);
    auto dW = torch::zeros({batch_dim, ctx_dim, 2*embed_dim}, torch::kCUDA);
    torch::Device device(torch::kCUDA);

    // Calculate SRAM size needed per block
    int tile_size = cache_dim * embed_dim / nhead;
    const int sram_size = (8 * tile_size * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(batch_dim * nhead, ctx_dim / cache_dim);
    dim3 block_dim(cache_dim);  // Bc threads per block

    backward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Y.data_ptr<float>(), 
        Ya.data_ptr<float>(), 
        W.data_ptr<float>(), 
        dY.data_ptr<float>(), 
        dYa.data_ptr<float>(),
        batch_dim, 
        ctx_dim, 
        embed_dim, 
        cache_dim, 
        activation, 
        alpha,
        nhead,
        X.data_ptr<float>(),
        Xa.data_ptr<float>(),
        dX.data_ptr<float>(), 
        dXa.data_ptr<float>(), 
        dW.data_ptr<float>(), 
        module_l
    );

    cudaDeviceSynchronize();

    return {X, Xa, dX, dXa, dW};
}
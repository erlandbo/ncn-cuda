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
    float* F_smem = &sram[tile_size * 2];
    float* dxj_smem = &sram[tile_size * 3];

    float* dY_smem = &sram[tile_size * 4];
    float* dYa_smem = &sram[tile_size * 5];
    float* dWi_smem = &sram[tile_size * 6];
    float* dWj_smem = &sram[tile_size * 7];

    const uint i = extract_group_index_simple(bcache, tx, cache_dim, ctx_dim);
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
            // Use reversibel trick xa
            Ya_smem[tx * nfeats + feat] = (Ya_smem[tx * nfeats + feat] - f * (1.0 - m)) / m;
            F_smem[tx * nfeats + feat] = f;
        }

        float dLdyi_dyidT_const = 0.0; 
        float dLdya_dyadT_const = 0.0; 
        
        for (int feat = 0; feat < nfeats; feat++) {
            float df_dt;
            if (activation == 0){
                df_dt = 1.0 - F_smem[tx * nfeats + feat] * F_smem[tx * nfeats + feat];
            }else if (activation == 1){
                df_dt = (F_smem[tx * nfeats + feat] > 0) ? 1.0 : 0.0;
            }else if (activation == 2){
                df_dt = F_smem[tx * nfeats + feat] * (1.0 - F_smem[tx * nfeats + feat]);
            }
            else {
                df_dt = 1.0;
            }

            float dyi_dt = (1.0 - m) * (1.0 - m) * df_dt;
            float dya_dt = (1.0 - m) * df_dt;
            
            dLdyi_dyidT_const += dY_smem[tx * nfeats + feat] * dyi_dt * Y_smem[j * nfeats + feat];
            dLdya_dyadT_const += dYa_smem[tx * nfeats + feat] * dya_dt * Y_smem[j * nfeats + feat];
        }

        // dxi from yi and ya
        for (int feat = 0; feat < nfeats; feat++) {
            float dWi = dLdyi_dyidT_const * Y_smem[tx * nfeats + feat] * (1.0-alpha) + \
                 dLdya_dyadT_const * Y_smem[tx * nfeats + feat] * (1.0-alpha);
            float dWj = dLdyi_dyidT_const * Y_smem[j * nfeats + feat] * (1.0-alpha) + \
                dLdya_dyadT_const * Y_smem[j * nfeats + feat] * (1.0-alpha);

            dWi_smem[tx * nfeats + feat] += dWi;
            dWj_smem[tx * nfeats + feat] += dWj;

            float df_dt;
            if (activation == 0){
                df_dt = 1.0 - F_smem[tx * nfeats + feat] * F_smem[tx * nfeats + feat];
            }else if (activation == 1){
                df_dt = (F_smem[tx * nfeats + feat] > 0) ? 1.0 : 0.0;
            }else if (activation == 2){
                df_dt = F_smem[tx * nfeats + feat] * (1.0 - F_smem[tx * nfeats + feat]);
            }
            else {
                df_dt = 1.0;
            }

            // dxi from yi and ya
            // dxa from yi and ya

            float dyi_dt = (1.0 - m) * (1.0 - m) * df_dt;
            float dya_dt = (1.0 - m) * df_dt;

            float dLdyi_dyidxi = dY_smem[tx * nfeats + feat] * m + alpha * dY_smem[tx * nfeats + feat] * dyi_dt + (1.0 - alpha) * dLdyi_dyidT_const * W[nfeats * bhead + feat]; 
            float dLdya_dyadxi = alpha * dYa_smem[tx * nfeats + feat] * dya_dt + (1.0 - alpha) * dLdya_dyadT_const * W[nfeats * bhead + feat]; 
            float dLdyi_dyidxa = dY_smem[tx * nfeats + feat] * m * (1.0 - m); 
            float dLdya_dyadxa = dYa_smem[tx * nfeats + feat] * m;

            float dxi = dLdyi_dyidxi + dLdya_dyadxi;
            float dxa = dLdyi_dyidxa + dLdya_dyadxa;

            // add dxj to dxi
            float dxj_i = (1.0-alpha) * dLdyi_dyidT_const * W[embed_dim + nfeats * bhead + feat] + (1.0-alpha) * dY_smem[tx * nfeats + feat] * dyi_dt * attn; 
            float dxj_a = (1.0-alpha) * dLdya_dyadT_const * W[embed_dim + nfeats * bhead + feat] + (1.0-alpha) * dYa_smem[tx * nfeats + feat] * dya_dt * attn; 
            float dxj = dxj_i + dxj_a;
            dxj_smem[tx * nfeats + feat] = dxj;
            
            __syncthreads();

            if (j == tx){
                dxi += dxj;
                for (int tj = 0; tj < cache_dim; tj++) {
                    if (tj != tx){
                        dxi += dxj_smem[tj * nfeats + feat];
                    }
                }            
            }
            dY_smem[tx * nfeats + feat] = dxi;
            dYa_smem[tx * nfeats + feat] = dxa;
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

    return {X, Xa, dX, dXa, dW};
}
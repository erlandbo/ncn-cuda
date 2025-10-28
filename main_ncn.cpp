#include <torch/extension.h>

std::vector< torch::Tensor > forward(
    torch::Tensor X, 
    torch::Tensor Xa, 
    torch::Tensor W, 
    const float alpha, 
    const int activation, 
    const int cache_dim, 
    const int nhead,
    const int module_l
);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}
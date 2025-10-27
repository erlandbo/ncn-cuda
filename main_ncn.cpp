#include <torch/extension.h>

std::tuple< torch::Tensor, torch::Tensor > forward(
    torch::Tensor X, 
    torch::Tensor Xa, 
    torch::Tensor W, 
    const float alpha, 
    const int activation, 
    const int n, 
    const int C,
    const int l
);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}
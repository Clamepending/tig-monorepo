// Lion optimizer kernel for c006.
//
// Lion (Chen et al. 2023, "Symbolic Discovery of Optimization Algorithms"):
//   c = beta1 * m + (1 - beta1) * g
//   update = -lr * sign(c) - lr * weight_decay * param
//   m = beta2 * m + (1 - beta2) * g
// Single-state (only momentum); simpler than Adam (which needs first + second
// moment), often comparable or better at well-tuned LR.

#include <stdint.h>

extern "C" __global__ void lion_step(
    const float* __restrict__ grad,
    const float* __restrict__ param,
    float* __restrict__ momentum,
    float* __restrict__ update,
    int n,
    float lr,
    float beta1,
    float beta2,
    float weight_decay
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float p = param[i];
    float m = momentum[i];

    float c = beta1 * m + (1.0f - beta1) * g;
    float s = (c > 0.0f) ? 1.0f : ((c < 0.0f) ? -1.0f : 0.0f);
    update[i] = -lr * (s + weight_decay * p);

    momentum[i] = beta2 * m + (1.0f - beta2) * g;
}

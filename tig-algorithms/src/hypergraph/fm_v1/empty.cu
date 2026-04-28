// Required by c005 build pipeline (.ptx is generated from a .cu file even
// when our algorithm doesn't actually invoke any custom GPU kernels).
// fm_v1 does its work CPU-side after a one-time GPU→host adjacency copy.

#include <stdint.h>

extern "C" __global__ void noop_kernel(int* x) {
    // No-op; never launched.
    if (x != 0 && false) { x[0] = 0; }
}

//-----------------------------------------------
// Even-odd preconditioning kernels on GPU
// (placeholder - the unpreconditioned solvers
// are the primary GPU code path)
//-----------------------------------------------
#include "gpu_state.cuh"
#include "colorspinor_device.cuh"

// Even-odd preconditioning is implemented via the
// hopping kernel in wilson_dirac_gpu.cu.
// The full preconditioned solver would call those
// kernels with parity filtering.
//
// For the initial GPU port, we use the unpreconditioned
// solvers (CG on normal equations, BiCGstab) which are
// already implemented. Even-odd preconditioning can be
// added as a performance optimization later.

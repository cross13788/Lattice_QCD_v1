#-----------------------------------------------
# Makefile for Lattice QCD
#
# Targets:
#   cpu  - Build with g++ (default)
#   gpu  - Build with nvc++ (host + CUDA)
#   clean - Remove object files and executables
#-----------------------------------------------
RUNDIR := ../run

#-----------------------------------------------
# CPU build settings (g++)
#-----------------------------------------------
CXX_CPU  := g++
CXXFLAGS_CPU = -std=c++17 -O3 -march=native -fopenmp -Wall \
               -Wno-unused-variable -Wno-unused-but-set-variable
LIBS_CPU = -lm

#-----------------------------------------------
# GPU build settings (nvc++ for everything)
#
# nvc++ has native CUDA support via -cuda flag.
# No nvcc needed — avoids GCC version issues.
# EXTRA_CXXFLAGS set by build.sh for --gcc-toolchain
# GPU_ARCH set by build.sh (default cc75 for Turing)
#-----------------------------------------------
EXTRA_CXXFLAGS ?=
GPU_ARCH       ?= cc75
CXX_GPU  := nvc++
CXXFLAGS_GPU = -std=c++17 -O3 -mp -Munroll -Mvect=simd -Mcache_align \
               -tp=native -Minfo=opt $(EXTRA_CXXFLAGS)
CUFLAGS      = -std=c++17 -O3 -mp -cuda -gpu=$(GPU_ARCH) -Munroll \
               -Mcache_align -tp=native -Minfo=accel,opt \
               --diag_suppress cuda_compile $(EXTRA_CXXFLAGS)
LIBS_GPU = -lm -lcudart

#-----------------------------------------------
# Shared sources (compiled for both CPU and GPU)
#-----------------------------------------------
SHARED_OBJECTS = constants_module.o       \
                 input_module.o           \
                 lattice_module.o         \
                 gamma_matrices.o         \
                 read_input_file.o        \
                 set_parameter.o          \
                 openmp_parallel_setup.o  \
                 initialize_gauge_field.o \
                 write_gauge_config.o     \
                 read_gauge_config.o      \
                 write_observables.o      \
                 write_correlators.o      \
                 jackknife_analysis.o     \
                 static_potential.o       \
                 effective_mass.o         \
                 point_source.o           \
                 measure_plaquette.o

#-----------------------------------------------
# CPU-only sources (replaced by GPU equivalents)
#-----------------------------------------------
CPU_ONLY_OBJECTS = su3_module.o              \
                   colorspinor_module.o      \
                   wilson_dirac_operator.o   \
                   wilson_dirac_dagger.o     \
                   cg_solver.o               \
                   bicgstab_solver.o         \
                   compute_staple.o          \
                   compute_plaquette.o       \
                   heat_bath_sweep.o         \
                   overrelaxation_sweep.o    \
                   metropolis_sweep.o        \
                   gauge_update_combined.o   \
                   thermalization.o          \
                   generate_configurations.o \
                   gauge_force.o             \
                   fermion_force.o           \
                   leapfrog_integrator.o     \
                   hmc_trajectory.o          \
                   hmc_driver.o              \
                   exp_su3_algebra.o         \
                   conjugate_momentum.o      \
                   generate_pseudofermion.o  \
                   pseudofermion_action.o    \
                   wilson_flow.o             \
                   wilson_loop.o             \
                   polyakov_loop.o           \
                   meson_correlator.o        \
                   compute_propagator.o      \
                   compute_action_density.o  \
                   clover_term.o             \
                   clover_dirac_operator.o   \
                   clover_even_odd.o         \
                   clover_fermion_force.o    \
                   even_odd_preconditioning.o \
                   gradient_check.o

CPU_MAIN = lqcd_main.o

#-----------------------------------------------
# GPU-only sources (CUDA kernels)
#-----------------------------------------------
GPU_CU_OBJECTS = gpu/gpu_state.o              \
                 gpu/gpu_setup.o              \
                 gpu/field_ops_gpu.o          \
                 gpu/gauge_ops_gpu.o          \
                 gpu/wilson_dirac_gpu.o       \
                 gpu/dirac_dispatch_gpu.o     \
                 gpu/cg_solver_gpu.o          \
                 gpu/bicgstab_solver_gpu.o    \
                 gpu/heat_bath_gpu.o          \
                 gpu/overrelaxation_gpu.o     \
                 gpu/gauge_update_gpu.o       \
                 gpu/su3_exp_gpu.o            \
                 gpu/gauge_force_gpu.o        \
                 gpu/fermion_force_gpu.o      \
                 gpu/random_fields_gpu.o      \
                 gpu/pseudofermion_action_gpu.o \
                 gpu/leapfrog_gpu.o           \
                 gpu/hmc_gpu.o                \
                 gpu/wilson_flow_gpu.o        \
                 gpu/clover_gpu.o             \
                 gpu/even_odd_gpu.o           \
                 gpu/measurements_gpu.o

# GPU main compiled with nvc++
GPU_MAIN = gpu/lqcd_main_gpu.o

# GPU also needs these CPU sources for host-side measurements
# Include all CPU sources needed for host-side operations.
# These get linked alongside the GPU objects so that
# host-side measurements, I/O, and fallback paths work.
GPU_HOST_OBJECTS = $(CPU_ONLY_OBJECTS)

#-----------------------------------------------
# Default target
#-----------------------------------------------
all: cpu

#-----------------------------------------------
# CPU target
#-----------------------------------------------
CPU_ALL_OBJECTS = $(SHARED_OBJECTS) $(CPU_ONLY_OBJECTS) $(CPU_MAIN)

cpu: $(RUNDIR)/lqcd.exe
	@echo ""
	@echo "CPU build complete: $(RUNDIR)/lqcd.exe"

$(RUNDIR)/lqcd.exe: $(CPU_ALL_OBJECTS)
	$(CXX_CPU) $(CXXFLAGS_CPU) -o $@ $^ $(LIBS_CPU)

#-----------------------------------------------
# GPU target
#-----------------------------------------------
GPU_ALL_OBJECTS = $(SHARED_OBJECTS) $(GPU_HOST_OBJECTS) $(GPU_CU_OBJECTS) $(GPU_MAIN)

gpu: gpu_shared gpu_host_objs gpu_cu_objs gpu_main_obj gpu_link
	@echo ""
	@echo "GPU build complete: $(RUNDIR)/lqcd_gpu.exe"

gpu_shared: $(SHARED_OBJECTS)

gpu_host_objs: $(GPU_HOST_OBJECTS)

gpu_cu_objs: $(GPU_CU_OBJECTS)

gpu_main_obj: $(GPU_MAIN)

gpu_link: $(GPU_ALL_OBJECTS)
	$(CXX_GPU) $(CUFLAGS) -o $(RUNDIR)/lqcd_gpu.exe $(GPU_ALL_OBJECTS) $(LIBS_GPU)

#-----------------------------------------------
# Compile rules
#-----------------------------------------------

# CPU: g++ for .cpp
%.o: %.cpp
	$(CXX_CPU) -c $(CXXFLAGS_CPU) $< -o $@

# GPU: nvc++ with -cuda for .cu files
gpu/%.o: gpu/%.cu
	$(CXX_GPU) -c $(CUFLAGS) $< -o $@

# GPU: nvc++ for GPU main .cpp
gpu/lqcd_main_gpu.o: gpu/lqcd_main_gpu.cpp
	$(CXX_GPU) -c $(CXXFLAGS_GPU) -DUSE_GPU $< -o $@

#-----------------------------------------------
# Module dependencies
#-----------------------------------------------
constants_module.o:

su3_module.o: constants_module.hpp

input_module.o: constants_module.hpp

lattice_module.o: constants_module.hpp input_module.hpp

colorspinor_module.o: constants_module.hpp

gamma_matrices.o: constants_module.hpp

#-----------------------------------------------
# Non-module dependencies
#-----------------------------------------------
read_input_file.o set_parameter.o openmp_parallel_setup.o: \
    constants_module.hpp input_module.hpp interfaces_module.hpp

initialize_gauge_field.o compute_plaquette.o compute_staple.o \
heat_bath_sweep.o overrelaxation_sweep.o metropolis_sweep.o \
gauge_update_combined.o thermalization.o generate_configurations.o \
measure_plaquette.o wilson_loop.o polyakov_loop.o static_potential.o \
write_gauge_config.o read_gauge_config.o write_observables.o \
jackknife_analysis.o wilson_flow.o compute_action_density.o: \
    constants_module.hpp input_module.hpp su3_module.hpp \
    lattice_module.hpp interfaces_module.hpp

wilson_dirac_operator.o wilson_dirac_dagger.o even_odd_preconditioning.o \
cg_solver.o bicgstab_solver.o point_source.o compute_propagator.o \
meson_correlator.o effective_mass.o write_correlators.o \
clover_term.o clover_dirac_operator.o clover_even_odd.o clover_fermion_force.o: \
    constants_module.hpp input_module.hpp su3_module.hpp \
    lattice_module.hpp colorspinor_module.hpp \
    gamma_matrices.hpp interfaces_module.hpp

clover_term.o clover_dirac_operator.o clover_even_odd.o clover_fermion_force.o: clover_module.hpp

exp_su3_algebra.o conjugate_momentum.o generate_pseudofermion.o \
pseudofermion_action.o gauge_force.o fermion_force.o \
leapfrog_integrator.o hmc_trajectory.o hmc_driver.o: \
    constants_module.hpp input_module.hpp su3_module.hpp \
    lattice_module.hpp colorspinor_module.hpp \
    gamma_matrices.hpp interfaces_module.hpp

lqcd_main.o: constants_module.hpp input_module.hpp su3_module.hpp \
             lattice_module.hpp colorspinor_module.hpp \
             gamma_matrices.hpp interfaces_module.hpp

#-----------------------------------------------
# Clean
#-----------------------------------------------
clean:
	rm -f *.o gpu/*.o $(RUNDIR)/lqcd.exe $(RUNDIR)/lqcd_gpu.exe
	@echo "Cleaned."

.PHONY: all cpu gpu clean gpu_shared gpu_host_objs gpu_cu_objs gpu_main_obj gpu_link

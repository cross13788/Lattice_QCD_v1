#-----------------------------------------------
# Makefile for Lattice QCD
#-----------------------------------------------
CXX := g++
RUNDIR := ../run

CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp -Wall \
           -Wno-unused-variable -Wno-unused-but-set-variable

LIBS = -lm

#-----------------------------------------------
# Object files in dependency order
#-----------------------------------------------
MODULE_OBJECTS = constants_module.o       \
                 su3_module.o             \
                 input_module.o           \
                 lattice_module.o         \
                 colorspinor_module.o     \
                 gamma_matrices.o

PHASE0_OBJECTS = read_input_file.o      \
                 set_parameter.o        \
                 openmp_parallel_setup.o

PHASE1_OBJECTS = initialize_gauge_field.o  \
                 compute_plaquette.o       \
                 compute_staple.o          \
                 heat_bath_sweep.o         \
                 overrelaxation_sweep.o    \
                 metropolis_sweep.o        \
                 gauge_update_combined.o   \
                 thermalization.o          \
                 generate_configurations.o \
                 measure_plaquette.o       \
                 wilson_loop.o             \
                 polyakov_loop.o           \
                 static_potential.o        \
                 write_gauge_config.o      \
                 read_gauge_config.o       \
                 write_observables.o       \
                 jackknife_analysis.o      \
                 wilson_flow.o

PHASE2_OBJECTS = wilson_dirac_operator.o    \
                 wilson_dirac_dagger.o      \
                 even_odd_preconditioning.o  \
                 cg_solver.o                \
                 bicgstab_solver.o           \
                 point_source.o             \
                 compute_propagator.o       \
                 meson_correlator.o         \
                 effective_mass.o           \
                 write_correlators.o        \
                 clover_term.o              \
                 clover_dirac_operator.o    \
                 clover_even_odd.o          \
                 clover_fermion_force.o

PHASE3_OBJECTS = exp_su3_algebra.o          \
                 conjugate_momentum.o       \
                 generate_pseudofermion.o    \
                 pseudofermion_action.o      \
                 gauge_force.o              \
                 fermion_force.o            \
                 leapfrog_integrator.o      \
                 hmc_trajectory.o           \
                 hmc_driver.o

VIZ_OBJECTS = compute_action_density.o

MAIN_OBJECT = lqcd_main.o

NOBJECTS = $(MODULE_OBJECTS) $(PHASE0_OBJECTS) $(PHASE1_OBJECTS) $(PHASE2_OBJECTS) $(PHASE3_OBJECTS) $(VIZ_OBJECTS) $(MAIN_OBJECT)

#-----------------------------------------------
# Build targets
#-----------------------------------------------
all: $(RUNDIR)/lqcd.exe
	@echo ""
	@echo "Build complete: $(RUNDIR)/lqcd.exe"

$(RUNDIR)/lqcd.exe: $(NOBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

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
# Non-module dependencies: all depend on core headers
#-----------------------------------------------
$(PHASE0_OBJECTS): constants_module.hpp input_module.hpp interfaces_module.hpp

$(PHASE1_OBJECTS): constants_module.hpp input_module.hpp su3_module.hpp \
                   lattice_module.hpp interfaces_module.hpp

$(PHASE2_OBJECTS): constants_module.hpp input_module.hpp su3_module.hpp \
                   lattice_module.hpp colorspinor_module.hpp \
                   gamma_matrices.hpp interfaces_module.hpp

clover_term.o clover_dirac_operator.o clover_even_odd.o clover_fermion_force.o: clover_module.hpp

$(PHASE3_OBJECTS): constants_module.hpp input_module.hpp su3_module.hpp \
                   lattice_module.hpp colorspinor_module.hpp \
                   gamma_matrices.hpp interfaces_module.hpp

$(VIZ_OBJECTS): constants_module.hpp input_module.hpp su3_module.hpp \
                lattice_module.hpp interfaces_module.hpp

$(MAIN_OBJECT): constants_module.hpp input_module.hpp su3_module.hpp \
                lattice_module.hpp colorspinor_module.hpp \
                gamma_matrices.hpp interfaces_module.hpp

#-----------------------------------------------
# Clean
#-----------------------------------------------
clean:
	rm -f *.o $(RUNDIR)/lqcd.exe
	@echo "Cleaned."

.PHONY: all clean

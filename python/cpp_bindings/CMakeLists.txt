cmake_minimum_required(VERSION 3.12)
project(bayesm_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug flags
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fno-omit-frame-pointer")

# Use specific Python from miniforge
set(Python_EXECUTABLE "/Users/carolh/miniforge3/bin/python")
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# pybind11 location
set(pybind11_DIR "/Users/carolh/miniforge3/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# Find Armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Include directories
include_directories(include)
include_directories(/opt/homebrew/include)

# Source files
set(SOURCES
    src/lndMvn_rcpp.cpp
    src/ghkvec_rcpp.cpp
    src/rwishart_rcpp.cpp
    src/rdirichlet_rcpp.cpp
    src/rtrun_rcpp.cpp
    src/lndIWishart_rcpp.cpp
    src/lndMvst_rcpp.cpp
    src/rmvst_rcpp.cpp
    src/breg_rcpp.cpp
    src/llmnl_rcpp.cpp
    src/lndIChisq_rcpp.cpp
    src/rmultireg_rcpp.cpp
    src/callroot_rcpp.cpp
    src/runireg_rcpp_loop.cpp
    src/runiregGibbs_rcpp_loop.cpp
    src/rhierLinearModel_rcpp_loop.cpp
    src/rmnlIndepMetrop_rcpp_loop.cpp
    src/rbprobitGibbs_rcpp_loop.cpp
    src/rordprobitGibbs_rcpp_loop.cpp
    src/rivGibbs_rcpp_loop.cpp
    src/rnegbinRw_rcpp_loop.cpp
    src/rsurGibbs_rcpp_loop.cpp
    src/rmvpGibbs_rcpp_loop.cpp
    src/rmnpGibbs_rcpp_loop.cpp
    src/rmixGibbs_rcpp.cpp
    src/rhierLinearMixture_rcpp_loop.cpp
    src/rhierMnlRwMixture_rcpp_loop.cpp
    src/rdpGibbs_rcpp_loop.cpp
    src/rivDP_rcpp_loop.cpp
    src/rhierMnlDP_rcpp_loop.cpp
    src/rbayesBLP_rcpp_loop.cpp
    src/rhierNegbinRw_rcpp_loop.cpp
    src/rscaleUsage_rcpp_loop.cpp
    bayesm_bindings.cpp
)

# Create Python module
pybind11_add_module(_bayesm_cpp ${SOURCES})

# Link Armadillo
target_link_libraries(_bayesm_cpp PRIVATE ${ARMADILLO_LIBRARIES})

# Install target
install(TARGETS _bayesm_cpp DESTINATION ../bayesm/_cpp)

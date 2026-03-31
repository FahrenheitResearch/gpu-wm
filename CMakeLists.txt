cmake_minimum_required(VERSION 3.18)
project(gpu-wm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Auto-detect GPU architecture
set(CMAKE_CUDA_ARCHITECTURES native)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

# Find NetCDF
find_path(NETCDF_INCLUDE_DIR netcdf.h)
find_library(NETCDF_LIBRARY netcdf)

# Find eccodes
find_path(ECCODES_INCLUDE_DIR eccodes.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include)
find_library(ECCODES_LIBRARY eccodes)

add_executable(gpu-wm
    src/main.cu
    src/core/dynamics.cu
    src/core/init.cu
    src/core/boundaries.cu
    src/core/sponge.cu
    src/physics/microphysics.cu
    src/physics/microphysics_thompson.cu
    src/physics/radiation.cu
    src/physics/pbl.cu
    src/io/output.cu
    src/io/netcdf_output.cu
    src/io/grib_reader.cu
)

target_include_directories(gpu-wm PRIVATE include)
# No cufft needed - split-explicit solver replaces FFT-based Poisson solver

# Precision control: float32 by default, float64 with -DUSE_DOUBLE=ON
option(USE_DOUBLE "Use double precision (float64) for 3D fields instead of float32" OFF)
if(USE_DOUBLE)
    message(STATUS "Precision: float64 (USE_DOUBLE=ON)")
    target_compile_definitions(gpu-wm PRIVATE USE_DOUBLE)
else()
    message(STATUS "Precision: float32 (default, pass -DUSE_DOUBLE=ON for float64)")
endif()

# NetCDF support
if(NETCDF_INCLUDE_DIR AND NETCDF_LIBRARY)
    message(STATUS "Found NetCDF: ${NETCDF_LIBRARY}")
    target_include_directories(gpu-wm PRIVATE ${NETCDF_INCLUDE_DIR})
    target_link_libraries(gpu-wm PRIVATE ${NETCDF_LIBRARY})
    target_compile_definitions(gpu-wm PRIVATE HAS_NETCDF)
else()
    message(WARNING "NetCDF not found - NetCDF output disabled")
endif()

# eccodes support
if(ECCODES_INCLUDE_DIR AND ECCODES_LIBRARY)
    message(STATUS "Found eccodes: ${ECCODES_LIBRARY}")
    target_include_directories(gpu-wm PRIVATE ${ECCODES_INCLUDE_DIR})
    target_link_libraries(gpu-wm PRIVATE ${ECCODES_LIBRARY})
    target_compile_definitions(gpu-wm PRIVATE HAS_ECCODES)
else()
    message(WARNING "eccodes not found - GRIB2 input disabled")
endif()

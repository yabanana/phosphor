# FindAftermathSDK.cmake — locate optional NVIDIA Aftermath SDK
# Expected layout:
#   third_party/aftermath/include/GFSDK_Aftermath_GpuCrashTracker.h
#   third_party/aftermath/lib/libGFSDK_Aftermath_Lib.x64.so

set(AFTERMATH_SEARCH_DIR "${CMAKE_SOURCE_DIR}/third_party/aftermath")

find_path(AFTERMATH_INCLUDE_DIR
    NAMES GFSDK_Aftermath_GpuCrashTracker.h
    PATHS "${AFTERMATH_SEARCH_DIR}/include"
    NO_DEFAULT_PATH
)

find_library(AFTERMATH_LIBRARY
    NAMES GFSDK_Aftermath_Lib.x64 GFSDK_Aftermath_Lib
    PATHS "${AFTERMATH_SEARCH_DIR}/lib"
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AftermathSDK
    DEFAULT_MSG
    AFTERMATH_INCLUDE_DIR
    AFTERMATH_LIBRARY
)

if(AftermathSDK_FOUND)
    set(AFTERMATH_FOUND TRUE)
else()
    set(AFTERMATH_FOUND FALSE)
endif()

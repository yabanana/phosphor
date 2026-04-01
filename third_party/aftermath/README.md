# NVIDIA Aftermath SDK

To enable GPU crash diagnostics, download the Aftermath SDK from:
https://developer.nvidia.com/nsight-aftermath

Place the files as follows:
```
aftermath/
├── include/
│   ├── GFSDK_Aftermath.h
│   ├── GFSDK_Aftermath_GpuCrashTracker.h
│   └── GFSDK_Aftermath_GpuCrashDump.h
└── lib/
    └── libGFSDK_Aftermath_Lib.x64.so
```

Then build with: `cmake --preset debug -DPHOSPHOR_ENABLE_AFTERMATH=ON`

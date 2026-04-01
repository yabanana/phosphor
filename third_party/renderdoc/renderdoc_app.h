// RenderDoc In-Application API header stub
// Download the full header from: https://github.com/baldurk/renderdoc/blob/v1.x/renderdoc/api/app/renderdoc_app.h
// This stub provides the minimal types needed for compilation when RenderDoc is not available.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum RENDERDOC_Version {
    eRENDERDOC_API_Version_1_6_0 = 10600,
} RENDERDOC_Version;

typedef enum RENDERDOC_CaptureOption {
    eRENDERDOC_Option_AllowVSync = 0,
    eRENDERDOC_Option_AllowFullscreen = 1,
    eRENDERDOC_Option_APIValidation = 2,
    eRENDERDOC_Option_CaptureCallstacks = 3,
    eRENDERDOC_Option_CaptureCallstacksOnlyActions = 4,
    eRENDERDOC_Option_DelayForDebugger = 5,
    eRENDERDOC_Option_VerifyBufferAccess = 6,
    eRENDERDOC_Option_HookIntoChildren = 7,
    eRENDERDOC_Option_RefAllResources = 8,
    eRENDERDOC_Option_SaveAllInitials = 9,
    eRENDERDOC_Option_CaptureAllCmdLists = 10,
    eRENDERDOC_Option_DebugOutputMute = 11,
    eRENDERDOC_Option_AllowUnsupportedVendorExtensions = 12,
} RENDERDOC_CaptureOption;

typedef void* RENDERDOC_DevicePointer;
typedef void* RENDERDOC_WindowHandle;

typedef struct RENDERDOC_API_1_6_0 {
    int (*GetAPIVersion)(int* major, int* minor, int* patch);
    int (*SetCaptureOptionU32)(RENDERDOC_CaptureOption opt, unsigned int val);
    int (*SetCaptureOptionF32)(RENDERDOC_CaptureOption opt, float val);
    unsigned int (*GetCaptureOptionU32)(RENDERDOC_CaptureOption opt);
    float (*GetCaptureOptionF32)(RENDERDOC_CaptureOption opt);
    void (*SetFocusToggleKeys)(unsigned int* keys, int num);
    void (*SetCaptureKeys)(unsigned int* keys, int num);
    unsigned int (*GetOverlayBits)();
    void (*MaskOverlayBits)(unsigned int And, unsigned int Or);
    void (*RemoveHooks)();
    void (*UnloadCrashHandler)();
    void (*SetCaptureFilePathTemplate)(const char* pathtemplate);
    const char* (*GetCaptureFilePathTemplate)();
    unsigned int (*GetNumCaptures)();
    unsigned int (*GetCapture)(unsigned int idx, char* filename, unsigned int* pathlength, unsigned long long* timestamp);
    void (*TriggerCapture)();
    unsigned int (*IsTargetControlConnected)();
    unsigned int (*LaunchReplayUI)(unsigned int connectTargetControl, const char* cmdline);
    void (*SetActiveWindow)(RENDERDOC_DevicePointer device, RENDERDOC_WindowHandle wndHandle);
    void (*StartFrameCapture)(RENDERDOC_DevicePointer device, RENDERDOC_WindowHandle wndHandle);
    unsigned int (*IsFrameCapturing)();
    unsigned int (*EndFrameCapture)(RENDERDOC_DevicePointer device, RENDERDOC_WindowHandle wndHandle);
    void (*TriggerMultiFrameCapture)(unsigned int numFrames);
    void (*SetCaptureFileComments)(const char* filePath, const char* comments);
    unsigned int (*DiscardFrameCapture)(RENDERDOC_DevicePointer device, RENDERDOC_WindowHandle wndHandle);
    unsigned int (*ShowReplayUI)();
    void (*SetCaptureTitle)(const char* title);
} RENDERDOC_API_1_6_0;

typedef int (*pRENDERDOC_GetAPI)(RENDERDOC_Version version, void** outAPIPointers);

#ifdef __cplusplus
}
#endif

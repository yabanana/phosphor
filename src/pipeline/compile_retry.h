#pragma once

#include "core/types.h"

#include <string>

namespace phosphor::pipe {

/// F3 robustness: under heavy machine load the macOS Metal compiler service
/// connection can drop; the compile then fails with an error whose text holds
/// "Compilation failed due to an interrupted connection:
/// XPC_ERROR_CONNECTION_INTERRUPTED" and the same call succeeds when repeated.
/// Only that exact condition is retried, a bounded number of times; real
/// compile errors are never retried.
inline constexpr u32 COMPILE_RETRY_LIMIT = 3;

inline constexpr const char* TRANSIENT_COMPILER_ERROR_TEXT =
    "Compilation failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED";

/// True for the transient compiler-service interruption (narrow match on the
/// error token, so a message that merely mentions a connection is not retried).
[[nodiscard]] inline bool isTransientCompilerInterruption(const std::string& description) {
    return description.find("XPC_ERROR_CONNECTION_INTERRUPTED") != std::string::npos;
}

/// Run `attempt(why)` (true = success; on failure it stores the error text in
/// `why`).  A failure that isTransientCompilerInterruption() is retried at most
/// `maxRetries` times; `onRetry(retryNumber, why)` runs before each retry (log,
/// count, back off).  Returns the final outcome; `why` holds the last error.
template <class Attempt, class OnRetry>
bool retryTransientCompile(u32 maxRetries, std::string& why, Attempt&& attempt, OnRetry&& onRetry) {
    for (u32 retry = 0;; ++retry) {
        why.clear();
        if (attempt(why)) return true;
        if (retry >= maxRetries || !isTransientCompilerInterruption(why)) return false;
        onRetry(retry + 1, why);
    }
}

} // namespace phosphor::pipe

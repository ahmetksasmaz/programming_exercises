#include "Logger.h"

#ifndef GpuErrChk
#define GpuErrChk(ans) { GpuAssert((ans), __FILE__, __LINE__); }
inline void GpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      LoggerPrint("GPU Assert: " + std::string{cudaGetErrorString(code)} + " " + file + " " + std::to_string(line) + "\n", LogLevel::FATAL);
   }
}
#endif
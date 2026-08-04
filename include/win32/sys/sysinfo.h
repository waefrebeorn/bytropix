/* include/win32/sys/sysinfo.h — minimal shim so #include <sys/sysinfo.h>
 * resolves on MSYS2/mingw64. Only the symbols wubu_affinity.c needs. */
#ifndef WUBU_WIN32_SYS_SYSINFO_H
#define WUBU_WIN32_SYS_SYSINFO_H

#include <windows.h>

struct sysinfo {
    long uptime;
    unsigned long loads[3];
    unsigned long totalram;
    unsigned long freeram;
    unsigned long sharedram;
    unsigned long bufferram;
    unsigned long totalswap;
    unsigned long freeswap;
    unsigned short procs;
    unsigned long totalhigh;
    unsigned long freehigh;
    unsigned int mem_unit;
};

static inline int get_nprocs(void) {
    SYSTEM_INFO si; GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
}

static inline int sysinfo(struct sysinfo *info) {
    if (!info) return -1;
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    info->totalram = (unsigned long)(ms.ullTotalPhys / 1024);
    info->freeram  = (unsigned long)(ms.ullAvailPhys / 1024);
    info->procs    = (unsigned short)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return 0;
}

#endif /* WUBU_WIN32_SYS_SYSINFO_H */

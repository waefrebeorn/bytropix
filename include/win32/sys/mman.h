/* include/win32/sys/mman.h — placeholder so #include <sys/mman.h> resolves on
 * MSYS2/mingw64 (which lacks this header). The actual mmap/munmap/msync shims
 * live in wubu_win.h (force-included by Makefile.win). This file exists only
 * to satisfy the #include line in the engine's .c files. */
#ifndef WUBU_WIN32_SYS_MMAN_H
#define WUBU_WIN32_SYS_MMAN_H
#include "wubu_win.h"
#endif

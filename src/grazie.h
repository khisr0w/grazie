/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  4/20/2023 6:22:08 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#if !defined(GRAZIE_H)

#if GRAZIE_PLT_WIN
#define _CRT_RAND_S /* NOTE(Abid): To use rand_s */
#include <windows.h>
#include <bcrypt.h>
#endif

#ifdef GRAZIE_PLT_LINUX
#include <sys/mman.h>
#include <stddef.h>
#endif

#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* NOTE(Abid): Unity includes */
#include "utils.h"
#include "gmath.c"
#include "rand.c"
#include "memory.c"
#include "tensor.c"
#include "autograd.c"
#include "loss.c"
#include "optimizer.c"
#include "module.c"

#define GRAZIE_H
#endif

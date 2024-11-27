/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  8/1/2024 7:53:55 PM                                           |
    |    Last Modified:                                                                |
    |                                                                                  |
    +======================================| Copyright © Sayed Abid Hashimi |==========+  */

#if !defined(MODULE_H)

typedef enum {
    module_None,
    module_Linear,
} module_type;

typedef struct {
    tensor_list weights;
    module_type type;
} module;

#define MODULE_H
#endif

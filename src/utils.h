/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  3/16/2023 6:25:57 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +=====================| Sayed Abid Hashimi, Copyright © All rights reserved |======+  */

#if !defined(UTILS_H)

#ifdef GRAZIE_DEBUG
#define Assert(Expr, ErrorStr) if(!(Expr)) { *(int32 *)0 = 0; }
#else
#define Assert(Expr, ErrorStr) if(!(Expr)) {fprintf(stderr, "ASSERTION ERROR (%s:%d): " ErrorStr "\nExiting...\n", __FILE__, __LINE__); exit(-1);}
#endif

#ifdef GRAZIE_DEBUG
#define InvalidCodePath *(int *)0 = 0
#else
#define InvalidCodePath do { fprintf(stderr, "Invalid Path (%s:%d): \nExiting...\n", __FILE__, __LINE__); exit(-1); } while(0)
#endif


#define ArrayLength(Array) (sizeof(Array)/sizeof(Array[0]))

// NOTE(Abid): Byte Macros
#define Kilobyte(Value) ((Value)*1024LL)
#define Megabyte(Value) (Kilobyte(Value)*1024LL)
#define Gigabyte(Value) (Megabyte(Value)*1024LL)
#define Terabyte(Value) (Gigabyte(Value)*1024LL)

// NOTE(Abid): typedef and static define for ease of use
typedef uint32_t uint32;
typedef int32_t int32;
typedef int64_t int64;
typedef uintptr_t uintptr;
typedef float float32;
typedef double float64;
typedef int8_t boolean;
#define internal static
#define local_persist static
#define global_var static
#define true 1
#define false 0

// NOTE(Abid): Defines if op history should be preserved for gradient calculation
#define GRAD_PRESERVE(Value) __GetSetGradState(true, Value)
#define GRAD_PRESERVE_TOGGLE() __GetSetGradState(true, !IS_GRAD_PRESERVE())
#define IS_GRAD_PRESERVE() (__GetSetGradState(false, 0))
internal inline
boolean __GetSetGradState(boolean Set, boolean NewState)
{
    local_persist boolean ShouldGrad = true;

    if(Set) ShouldGrad = NewState;

    return ShouldGrad;
}

#define UTILS_H
#endif

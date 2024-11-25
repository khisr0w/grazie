/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  9/18/2024 7:07:52 PM                                          |
    |    Last Modified:                                                                |
    |                                                                                  |
    +======================================| Copyright © Sayed Abid Hashimi |==========+  */

internal inline f32
gz_clamp(f32 Value, f32 Min, f32 Max) {
    f32 ClampBelow = Value < Min ? Min : Value;
    return ClampBelow > Max ? Max : ClampBelow;
}

internal inline f32 gz_logf(f32 value) { return logf(value); }
internal inline f64 gz_log(f64 value) { return log(value); }



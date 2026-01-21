# 在Fedora43使用CUDA13
在Fedora43上使用cuda13时,遇到
``` bash
/usr/include/bits/mathcalls.h(206): error: exception specification is incompatible with that of previous function "rsqrtf" (declared at line 653 of /usr/local/cuda/include/crt/math_functions.h)
  extern float rsqrtf (float __x) noexcept (true); extern float __rsqrtf (float __x) noexcept (true);
``` 
将`/usr/local/cuda/include/crt/math_functions.h`相对应的行数的定义加上`noexccept(true);`
如
``` bash
extern float __rsqrtf (float __x)
```
变为
``` bash
extern float __rsqrtf (float __x) noexcept(true);
```


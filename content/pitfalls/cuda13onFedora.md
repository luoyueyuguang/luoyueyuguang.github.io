在 Fedora 43 上用 CUDA 13.0/13.1（默认 GCC 15）编译时，遇到

```bash
/usr/include/bits/mathcalls.h(206): error: exception specification is incompatible with that of previous function "rsqrt" (declared at line 629 of /usr/local/cuda/include/crt/math_functions.h)
  extern double rsqrt (double __x) noexcept (true); extern double __rsqrt (double __x) noexcept (true);
/usr/include/bits/mathcalls.h(206): error: exception specification is incompatible with that of previous function "rsqrtf" (declared at line 653 of /usr/local/cuda/include/crt/math_functions.h)
  extern float rsqrtf (float __x) noexcept (true); extern float __rsqrtf (float __x) noexcept (true);
```

glibc 的 `<math.h>` 用 `noexcept(true)` 声明 `rsqrt`/`rsqrtf`，而 CUDA 13 的 `crt/math_functions.h` 没加，两者冲突，因为 CUDA 13 尚未支持 GCC 15（Fedora 43 默认编译器）。

省事的方案是用受支持的 GCC 13 编译，别动 CUDA 头文件：

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
```

硬改头文件则给 `rsqrt` 和 `rsqrtf` 两个公开声明都补 `noexcept (true)`（改的是 `rsqrt`/`rsqrtf`，不是内部的 `__rsqrt`/`__rsqrtf`）；CUDA 13.2 起已修复，13.3 官方支持 Fedora 44（GCC 15.2 / glibc 2.43）。

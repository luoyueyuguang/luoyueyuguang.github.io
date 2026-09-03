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

截至 2026-09，最新 CUDA 仍是 13.3（含 13.3 Update 1）；NVIDIA 为 Fedora 44 认证的是「默认 GCC 15.2.1 / glibc 2.43 / 内核 6.19.10-300」。但这并不等于 Fedora 44 现在的默认 GCC 还是 15——这台机器上默认 `gcc` 已经升到 16.2.1（`gcc (GCC) 16.2.1 20260819`），而 CUDA 13.3 支持的宿主 GCC 区间是 6.x–15.x，16.x 落在区间之外。所以「换一个受支持的旧 GCC」这条建议至今仍然成立：在 Fedora 44 上同样把 `-DCMAKE_CUDA_HOST_COMPILER` 指向 `g++-15`/`g++-13` 这类 6.x–15.x 内的编译器即可。对你这台机器（Fedora 44 + CUDA 13.1 + GCC 16.2.1）来说，最省事的修法是把 CUDA 升到 13.2 以上——`rsqrt`/`rsqrtf` 的 `noexcept` 硬错误随之消失，也就不必再换 GCC。

## Reference

- [CUDA Installation Guide for Linux（CUDA 13.3）](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) —「Native Linux Distribution Support」表：Fedora 44 默认 GCC 15.2.1 / glibc 2.43 / 内核 6.19.10-300；「Supported Compilers」表：x86_64 支持 GCC 6.x–15.x。
- [CUDA Toolkit 13.3 Update 1 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

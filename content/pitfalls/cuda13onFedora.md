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

硬改头文件则给 `rsqrt` 和 `rsqrtf` 两个公开声明都补 `noexcept (true)`（改的是 `rsqrt`/`rsqrtf`，不是内部的 `__rsqrt`/`__rsqrtf`，`cospi`/`sinpi` 那几个 13.x 里已经用 `__NV_IEC_60559_FUNCS_EXCEPTION_SPECIFIER` 兜住了）；这个问题在 CUDA 13.2 里被修掉。

截至 2026-09，最新 CUDA 已是 13.4（GA 版本号 13.4.1，取代 13.4.0 developer preview）。13.3 时为 Fedora 44 认证的是「默认 GCC 15.2.1 / glibc 2.43 / 内核 6.19.10-300」，宿主 GCC 支持区间 6.x–15.x；13.4 这张表换成「默认 GCC 16.0.1 / glibc 2.43 / 内核 6.19.2-300」，GCC 上界也抬到 16.x。所以这台机器上默认的 `gcc 16.2.1`（`gcc (GCC) 16.2.1 20260819`）在 13.3 及更早版本里落在支持区间外，在 13.4 里已经在区间内。对这台机器（Fedora 44 + CUDA 13.1 + GCC 16.2.1）来说，最省事的修法是把 CUDA 升到 13.2 以上——`rsqrt`/`rsqrtf` 的 `noexcept` 硬错误随之消失；升到 13.4 连 GCC 16 都不用再换。留在 13.3 及更早版本时，「换一个受支持的旧 GCC」这条建议继续成立：把 `-DCMAKE_CUDA_HOST_COMPILER` 指向 `g++-15`/`g++-13` 这类 6.x–15.x 内的编译器即可。

## Reference

- [CUDA Installation Guide for Linux（CUDA 13.4）](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) —「Native Linux Distribution Support」表：Fedora 44 默认 GCC 16.0.1 / glibc 2.43 / 内核 6.19.2-300；「Supported Compilers」表：x86_64 支持 GCC 6.x–16.x。
- [CUDA Installation Guide for Linux（CUDA 13.3，归档）](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-installation-guide-linux/index.html) — 同上两张表在 13.3 的取值：Fedora 44 默认 GCC 15.2.1 / 内核 6.19.10-300，宿主 GCC 6.x–15.x。
- [CUDA Toolkit 13.4 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) — 13.4 GA 即 13.4.1。
- [NVIDIA 开发者论坛：Fedora 43 与 CUDA 13.1 的 rsqrt/rsqrtf 报错](https://forums.developer.nvidia.com/t/fedora-43-and-nvcc-cuda13-1-error-exception-specification-is-incompatible-rsqrt-rsqrtf/354510) — 报错文本与头文件补丁的出处；回帖确认已在 CUDA 13.2 修复。

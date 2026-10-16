---
aiGenerated: true
---

晚上想踢两脚 eFootball。图标点下去，资源加载走完、KONAMI 的商标转过去，就在要进游戏界面的那一下回到桌面。每次都是同一个位置。手机是 Redmi K50，HyperOS 2.0（Android 14），游戏是 Play 商店装的 `jp.konami.pesam` 11.1.1。

我决定把它查清楚，而不是重启一遍手机再试。下面是整个过程，包括一次白干的复现和一条被我否掉的错误方向。

## 第一道墙：手机根本没和电脑说话

```bash
$ adb devices -l
List of devices attached
```

空的。adb 服务本身没问题（重启过），所以问题在设备侧。看 USB 描述符：

```bash
$ lsusb -d 2717:ff40 -v | grep -E "bNumInterfaces|bInterfaceClass|bInterfaceSubClass|bInterfaceProtocol|iInterface"
    bNumInterfaces          1
      bInterfaceClass         6 Imaging
      bInterfaceSubClass      1 Still Image Capture
      bInterfaceProtocol      1 Picture Transfer Protocol (PIMA 15470)
      iInterface              5 MTP
```

整机只暴露了一个接口，`class 6 / 1 / 1`，就是 MTP。手机开着 USB 调试时会出现第二个接口 `class ff / 42 / 01`、`iInterface` 写着 `ADB`。这里没有，说明 USB 调试压根没开，不是 udev 权限问题（权限不足会显示 `no permissions` 而不是空列表）。

小米这边要多点两个开关，`开发者选项 → USB 调试`，加上 MIUI 特有的 `USB 调试（安全设置）`。开完插拔一次数据线，手机弹授权框：

```bash
$ adb devices -l
ZPAYINDIAUNZFULN       unauthorized usb:3-1 transport_id:1
$ adb devices -l
ZPAYINDIAUNZFULN       device usb:3-1 product:rubens model:22041211AC device:rubens transport_id:1
```

第一次显示 `unauthorized`，把 adb 服务重启一遍让授权框重新弹，点了「始终允许」之后才拿到 `device`。

## 第一次复现是废的

我的第一版复现脚本，清空日志、强停游戏、启动、盯着 `pidof`。进程 45 秒后消失，日志里也确实有死亡记录。看起来是个干净的复现。

差一点就信了。往下翻两行：

```text
10-16 23:02:01.605  1679  1768 I am_kill : [0,8765,jp.konami.pesam,905,stop jp.konami.pesam due to SwipeUpClean,792304]
10-16 23:02:01.621   919   919 I Zygote  : Process 8765 exited due to signal 9 (Killed)
```

`SwipeUpClean`。这是「上滑清理」，我自己在复现期间手滑把游戏划掉了，顺带把它判成了崩溃。死亡时间往前翻，还能看到 `EnterRecentFromAppEvent` 和 `onForegroundInfoChanged` 把前台切走。

重来一遍。这次我不碰屏幕，每 3 秒截一张图，只观察：

```text
[0.0s] pid=10031
[106.8s] CONFIRMED EXITED
```

106 秒，无人触碰。再跑一次，37 秒。再跑，101 秒。每次都死，时间不固定，但都在进游戏的那几秒内。

顺手排掉一个方向：小米的游戏加速、Game Center 这些有没有往进程里塞东西。把 `com.xiaomi.gamecenter` 和 `com.xiaomi.gamecenter.sdk.service` 拉下来搜符号，一个都没命中。不是注入。

## 墓碑

`logcat -b crash` 里有完整的 native 墓碑，这是全部线索所在：

```text
10-16 23:04:40.053 10031 10130 F libc    : Fatal signal 6 (SIGABRT), code -1 (SI_QUEUE) in tid 10130 (GameThread), pid 10031 (jp.konami.pesam)
10-16 23:04:40.609 10587 10587 F DEBUG   : Cmdline: jp.konami.pesam
10-16 23:04:40.609 10587 10587 F DEBUG   : pid: 10031, tid: 10130, name: GameThread  >>> jp.konami.pesam <<<
10-16 23:04:40.609 10587 10587 F DEBUG   : signal 6 (SIGABRT), code -1 (SI_QUEUE), fault addr --------
10-16 23:04:40.609 10587 10587 F DEBUG   : Abort message: 'No pending exception expected: java.lang.NoSuchMethodError: no non-static method "Lcom/epicgames/ue4/GameActivity;.getItemDetails(Ljava/lang/String;[Ljava/lang/String;)V"
10-16 23:04:40.609 10587 10587 F DEBUG   : (Throwable with empty stack trace)'
```

回栈把它讲完了：

```text
#04 pc 006b54e8  /apex/com.android.art/lib64/libart.so (art::Thread::AssertNoPendingException() const+1576)
#05 pc 0027c620  /apex/com.android.art/lib64/libart.so (art::ClassLinker::FindClass(...)+64)
#06 pc 00478744  /apex/com.android.art/lib64/libart.so (art::JNI<false>::FindClass(_JNIEnv*, char const*)+1204)
#07 pc 0000000002f4f5a8  .../lib/arm64/libUE4.so (BuildId: 9068c18632a8023ad3e9c6faf3fe92d9)
#08 pc 000000000821ffc0  .../lib/arm64/libUE4.so
...（中间 20 多帧全是 libUE4.so）
#36 pc 0000000000101d5c  /apex/com.android.runtime/lib64/bionic/libc.so (__pthread_start(void*)+204)
```

读法是自下而上：`GameThread` 上跑的是 libUE4.so 的代码，它调 JNI `FindClass`，`FindClass` 进去发现自己这线程上还挂着一个未处理的异常，ART 直接 abort。那个待决异常是 `NoSuchMethodError`，说明在这之前还有一次失败的 JNI 调用（`CallObjectMethod`），返回值没人检查，异常也没人清。

被找的方法是 `com.epicgames.ue4.GameActivity.getItemDetails(String, String[])`。

## 证明这个方法真的不在

接下来要确认的不是「游戏崩了」，而是「libUE4.so 在找一个自己的 APK 里不存在的方法」。

主机上没有 jadx、baksmali、apktool，androguard 也没装（`ModuleNotFoundError`）。`dexdump` 也没有。与其装一套工具链，我直接写了个解析器读 dex 头：`string_ids`、`type_ids`、`proto_ids`、`method_ids`、`class_defs` 几张表按偏移取，`class_data_item` 里的 ULEB128 拿方法列表。

先把字符串层面的结论拿到，这个最快也最硬：

```bash
$ for f in classes.dex classes2.dex classes3.dex lib/arm64-v8a/libUE4.so; do
    printf "%-24s %s\n" "$f" "$(strings -a "$f" | grep -cx getItemDetails)"
  done
classes.dex              0
classes2.dex             0
classes3.dex             0
lib/arm64-v8a/libUE4.so  1
```

`getItemDetails` 这个标识符在三个 dex 里一次都没出现，只出现在 libUE4.so 里。大小写不敏感地再扫一遍 dex，命中的只有 `OnGetItemDetailsFinishedListener`、`onGetItemDetailsFinished`、`nativeOnGetItemDetailsFinished` 三个名字，仍然没有 `getItemDetails` 本体。

解析器再确认方法表：

```text
classes.dex    strings 56331  methods 64177  classes 8121
classes2.dex   strings 38024  methods 65129  classes 10906
classes3.dex   strings 36425  methods 32975  classes 4549

### Lcom/epicgames/ue4/GameActivity; in classes.dex: 218 methods
    virtual AndroidThunkJava_IapBeginPurchase (Ljava/lang/String;Ljava/lang/String;)Z
    virtual AndroidThunkJava_IapQueryInAppPurchases ([Ljava/lang/String;)Z
    virtual AndroidThunkJava_IapRestorePurchases ([Ljava/lang/String;[Z)Z
    virtual AndroidThunkJava_IapSetupService (Ljava/lang/String;)V
    ...（218 个，没有 getItemDetails）
```

`GameActivity` 有 218 个方法，一堆计费相关，就是没有 native 要找的那一个。所以这不是「手机上装错了版本」，是 APK 内部自己和自己对不上：native 段和 dex 段在编译时就没同步。

## 版本漂移的指纹

`libUE4.so` 里 `getItemDetails` 这个字符串的前后邻居很有意思：

```text
 ad1e0a getIntroductoryPriceCycles
 ad1e25 getItemDetails
 ad1e34 EMPTY_RECEIPT
 ad1e42 authorization_code
```

一水儿的计费 API 名字。再往下找 KONAMI 自己那套 IAB 框架，签名对上了：

```text
### Ljp/konami/android/common/iab/KonamiIabBaseNativeActivity;
    virtual getItemProductDetails (Ljava/lang/String;[Ljava/lang/String;)V
    virtual nativeOnGetItemDetailsFinished (Lcom/android/billingclient/api/BillingResult;Ljava/util/List;)V
    virtual nativeOnGetItemProductDetailsFinished (Lcom/android/billingclient/api/BillingResult;Ljava/util/List;)V
```

签名 `(Ljava/lang/String;[Ljava/lang/String;)V` 一模一样，只是名字从 `getItemDetails` 变成了 `getItemProductDetails`。而 libUE4.so 里两套命名都在（`getItemDetails`、`getItemProductDetails`、`nativeOnGetItemDetailsFinished`、`nativeOnGetItemProductDetailsFinished` 四个字符串齐全），dex 里只有新的那套。

KONAMI 在某个版本把接口改名，重编了 Java 层，native 库里那条旧调用没跟着更新。

## 为什么只在某些机器上炸

崩溃位置在计费初始化，那就去看计费。系统进程在崩溃前一直在刷：

```text
10-16 23:03:02.577 27237  9067 W Finsky  : [147] hgy.h(10): Unsupported billing API version: 27
10-16 23:03:02.580 27237  9067 W Finsky  : [147] hgy.h(10): Unsupported billing API version: 26
...
10-16 23:05:07.399 27237 10960 W Finsky  : [166] hgy.h(10): Unsupported billing API version: 17
```

Play 计费接口从 27 一路探测到 17，全被拒。去看这台机器上的 Play 商店：

```text
  Package [com.android.vending]:
    codePath=/data/app/~~HPr8zz0OSQAt3sVEgl_FuA==/com.android.vending-7BLpMDV8r3QqvQGH3heVhg==
    versionCode=82582010 minSdk=21 targetSdk=30
    versionName=25.8.20-21 [0] [PR] 379052828
    lastUpdateTime=2022-06-19 23:15:58
    installerPackageName=com.xiaomi.market
    pkgFlags=[ SYSTEM HAS_CODE ALLOW_CLEAR_USER_DATA UPDATED_SYSTEM_APP ALLOW_BACKUP RESTORE_ANY_VERSION ]
```

2022 年 6 月的 25.8.20，装的来源是小米商店。这个版本的计费 API 停在 19 以下，`ProductDetails` 那一代接口没有，所以游戏切到了旧路径，而旧路径正是那条名字错配的调用。

链路到此闭合：

1. Play 商店版本过旧，计费 API 版本被拒
2. 游戏回落到遗留计费分支
3. 遗留分支在 libUE4.so 里调一个 dex 里不存在的 Java 方法
4. JNI 调用失败留下未清异常，native 接着调 `FindClass`
5. ART 的 `AssertNoPendingException` 触发 SIGABRT

第 3 步是游戏自己的 bug，第 1 步是这台机器的状态。两者都得在，才会崩。

## 修

先试官方通道。Play 商店设置里点版本号，它自己说 `Google Play Store is up to date`，版本号纹丝不动，MIUI 这边把更新钉住了。小米商店搜不到 Google Play。

那就换 Play 商店本体的 APK。从 APKMirror 取 53.5.48 的 Android 14 变体，是个 bundle，base APK 加 39 个 split，我只下必需的：base、`config.arm64_v8a`、`config.zh`、`config.en`、两个 `phonesky_data_loader`、两个 `phonesky_webrtc_native_lib`。安装：

```bash
$ adb install-multiple -r --user 0 \
    base.apk split_config.arm64_v8a.apk split_config.zh.apk split_config.en.apk \
    split_phonesky_data_loader.apk split_phonesky_data_loader.config.arm64_v8a.apk \
    split_phonesky_webrtc_native_lib.apk split_phonesky_webrtc_native_lib.config.arm64_v8a.apk
Success
```

替换系统更新过的应用要签名一致才装得上，这里签名是 Google 原签，直接过了。装之前把旧版 `base.apk` 备份到本地，需要回滚就用得上。

## 结果

同一个复现脚本再跑一遍：

```text
vending now: versionName=53.5.48-34 [0] [PR] 995634172
launch: ok
[0.0s] pid=13234
SUMMARY: first= (0.0, '13234')  died= None  elapsed= 201.5
crash-buffer hits: 0
```

201.5 秒没退出，崩溃缓冲区里 `getItemDetails` 零条，`SIGABRT` 零条。之前每次必崩的位置是 35 到 107 秒。截图看进去，启动画面已经过去，进程停在游戏内的界面上。这个进程后来跨过一次 USB 掉线重连，PID 没变（17118），仍在以 30fps 出帧。

Play 商店升完第一次启动报了一次「出了点问题」，第二次进去内容正常，账号和购买记录都在。

## 边界

这条不是修好了游戏的 bug，是绕开了它的触发条件。`libUE4.so` 里那个错名字还在，只要计费再回落到旧分支就还会炸。真要根治得等 KONAMI 重编 native 层，他们发修复版本之后，把 Play 商店恢复成官方渠道更新就行。

反过来说，Play 商店版本正常的机器根本碰不到这条路径，所以这类崩溃报不到开发商那里：它在出厂系统镜像偏旧的机器上高发，在开发者自己的机器上不复现。下次遇到「一个应用只有某批设备闪退」，先看这类版本门控的分支，比盯着应用自己的日志快。

顺带记两条这次用上的手法。一是判断进程消失是真崩还是被杀，必须看 `am_kill` 和 `am_proc_died` 的 reason，`SwipeUpClean` 和 `lowmemorykiller` 都会伪装成崩溃。二是 native 与 Java 对不上时，`strings` 交叉比对一个符号在 `.so` 和 `dex` 里的存在性，比反编译整个 APK 快得多，也足够定性。

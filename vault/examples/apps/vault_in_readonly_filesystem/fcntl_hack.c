#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <errno.h>
#include <stdarg.h>
#include <string.h>

/*
 *  我们自己实现的 flock 函数，这是 Python fcntl.flock 的直接目标
 */
int flock(int fd, int operation) {
    // 定义一个静态函数指针，用于保存原始的 flock 函数地址
    // 'static' 确保它只被初始化一次
    static int (*original_flock)(int, int) = NULL;

    // 首次调用时，通过 dlsym 查找并保存原始 flock 函数的地址
    if (!original_flock) {
        original_flock = dlsym(RTLD_NEXT, "flock");
        // 如果 dlsym 查找失败，打印错误并退出
        if (!original_flock) {
            fprintf(stderr, "Error in dlsym for flock: %s\n", dlerror());
            return -1;
        }
    }

    // 调用原始的 flock 函数，并保存其返回值和 errno
    int result = original_flock(fd, operation);
    int original_errno = errno;

    // ★★★ 核心 hack 逻辑 ★★★
    // 检查是否是调用失败，并且错误码是我们想要拦截的 EROFS
    if (result == -1 && original_errno == EROFS) {
        // 确认这是一个加锁操作
        if ((operation & LOCK_SH) || (operation & LOCK_EX)) {
             errno = ENOTSUP; // 将 EROFS (30) 篡改为 ENOTSUP (95)
        }
    }
    // 注意：这里不需要 'else' 分支来恢复 errno，因为如果条件不满足，
    // original_errno 从未被修改，它仍然是原始调用的结果。

    return result;
}

/*
 *  为了覆盖更广的场景（例如 DuckDB 内部可能使用 fcntl 实现锁），
 *  我们保留对 fcntl 的劫持。逻辑与 flock 类似。
 */
int fcntl(int fd, int cmd, ...) {
    static int (*original_fcntl)(int fd, int cmd, ...);

    if (!original_fcntl) {
        original_fcntl = dlsym(RTLD_NEXT, "fcntl");
        if (!original_fcntl) {
            fprintf(stderr, "Error in dlsym for fcntl: %s\n", dlerror());
            return -1;
        }
    }

    va_list args;
    va_start(args, cmd);
    void *argp = va_arg(args, void *);
    va_end(args);

    int result = original_fcntl(fd, cmd, argp);
    int original_errno = errno;

    
    // 检查是否是加锁命令 (F_SETLK/F_SETLKW) 且返回了 EROFS
    if (result == -1 && original_errno == EROFS) {
        if (cmd == F_SETLK || cmd == F_SETLKW) {
             errno = ENOTSUP;
        }
    }

    return result;
}
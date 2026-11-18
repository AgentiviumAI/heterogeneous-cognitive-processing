"""Trusted direct-process runner. Fail closed if seccomp cannot be installed."""
import ctypes
import errno
import json
import os
import resource
import sys

payload = json.load(sys.stdin)
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
lib = ctypes.CDLL('libseccomp.so.2', use_errno=True)
lib.seccomp_init.argtypes = [ctypes.c_uint32]
lib.seccomp_init.restype = ctypes.c_void_p
lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
lib.seccomp_load.argtypes = [ctypes.c_void_p]
class Compare(ctypes.Structure):
    _fields_ = [('arg', ctypes.c_uint), ('op', ctypes.c_uint), ('a', ctypes.c_uint64), ('b', ctypes.c_uint64)]
ctx = lib.seccomp_init(0x7fff0000)
if not ctx:
    raise RuntimeError('seccomp init failed')
blocked = 'socket socketpair connect bind listen accept accept4 sendto sendmsg recvfrom recvmsg execve execveat fork vfork clone clone3 ptrace kill tkill tgkill mount umount2 unlink unlinkat rename renameat renameat2 mkdir mkdirat rmdir link linkat symlink symlinkat chmod fchmod fchmodat chown fchown fchownat truncate ftruncate creat openat2 io_uring_setup process_vm_writev'.split()
for name in blocked:
    number = lib.seccomp_syscall_resolve_name(name.encode())
    if number >= 0 and lib.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, number, 0) != 0:
        raise RuntimeError('seccomp rule failed: ' + name)
for syscall, arg in [('open', 1), ('openat', 2)]:
    number = lib.seccomp_syscall_resolve_name(syscall.encode())
    for flag in (os.O_WRONLY, os.O_RDWR, os.O_CREAT, os.O_TRUNC):
        if lib.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, number, 1, Compare(arg, 7, flag, flag)) != 0:
            raise RuntimeError('seccomp write rule failed')
if os.geteuid() == 0:
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
if lib.seccomp_load(ctx) != 0:
    raise RuntimeError('seccomp load failed')
namespace = {'__name__': '__main__'}
exec(compile(payload['setup'] + '\n' + payload['code'], '<solution>', 'exec'), namespace)
for test in payload['tests']:
    exec(compile(test, '<test>', 'exec'), namespace)
print('HCP_ALL_TESTS_PASS')

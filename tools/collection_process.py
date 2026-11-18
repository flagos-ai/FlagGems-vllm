# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded collector subprocesses without changing enclosing process groups."""
import ctypes
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def _children(parents):
    if sys.platform == "linux":
        children = set()
        for pid in parents:
            try:
                for path in Path(f"/proc/{pid}/task").glob("*/children"):
                    try:
                        children.update(map(int, path.read_text().split()))
                    except FileNotFoundError:
                        pass
            except FileNotFoundError:
                pass
        return children
    # NVIDIA CUDA collection targets Linux. Other POSIX hosts retain the older
    # tree cleanup, but cannot recover descendants already reparented to PID 1.
    table = subprocess.check_output(
        ["/bin/ps", "-eo", "pid=,ppid="], text=True, timeout=5
    )
    return {
        int(pid)
        for line in table.splitlines()
        for pid, ppid in [line.split()]
        if int(ppid) in parents
    }


def _kill_tree(root):
    """Freeze the owned subtree before discovery, then kill leaves first."""
    stopped = []
    pending = {root}
    try:
        while pending:
            for pid in pending:
                try:
                    os.kill(pid, signal.SIGSTOP)
                except ProcessLookupError:
                    continue
                stopped.append(pid)
            pending = _children(set(stopped)) - set(stopped)
    finally:
        for pid in reversed(stopped):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _supervise(error_fd, command):
    """Adopt orphaned Linux descendants while staying in the caller's group.

    Subreaper state belongs to this short-lived supervisor, never to the Python
    collector (which may run multiple commands in different threads).
    """
    # Ctrl-C also reaches this process. Keep the adoption boundary alive while
    # the collector handles KeyboardInterrupt and cleans up. Caught handlers are
    # reset on exec, so the command still receives its normal SIGINT behavior.
    signal.signal(signal.SIGINT, lambda *_: None)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        process = subprocess.Popen(command)
    except OSError as exc:
        os.write(error_fd, json.dumps([exc.errno, exc.strerror, exc.filename]).encode())
        os.close(error_fd)
        return 127
    os.close(error_fd)
    result = process.wait()
    # Keep the adoption boundary alive until every descendant has exited. In
    # particular, an exited shell must not release a still-running NCU to PID 1.
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            return result


def run_with_cleanup(command, *, timeout, capture_output=False, text=False, env=None):
    """Match collector run() calls; Linux also owns reparented descendants."""
    pipe = subprocess.PIPE if capture_output else None
    read_fd = write_fd = None
    actual_command = command
    options = {}
    if env is not None:
        options["env"] = env
    if sys.platform == "linux":
        read_fd, write_fd = os.pipe()
        actual_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(write_fd),
            *command,
        ]
        options["pass_fds"] = (write_fd,)
    try:
        with subprocess.Popen(
            actual_command, stdout=pipe, stderr=pipe, text=text, **options
        ) as process:
            if write_fd is not None:
                os.close(write_fd)
                write_fd = None
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except BaseException as exc:
                try:
                    _kill_tree(process.pid)
                finally:
                    process.wait()
                if isinstance(exc, subprocess.TimeoutExpired):
                    exc.cmd = command
                raise
            if read_fd is not None:
                error = os.read(read_fd, 65536)
                if error:
                    raise OSError(*json.loads(error))
            return subprocess.CompletedProcess(
                command, process.returncode, stdout, stderr
            )
    finally:
        for fd in (read_fd, write_fd):
            if fd is not None:
                os.close(fd)


if __name__ == "__main__":
    result = _supervise(int(sys.argv[1]), sys.argv[2:])
    if result < 0:
        # Preserve Popen's negative returncode for commands terminated by signal.
        if -result not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(-result, signal.SIG_DFL)
        os.kill(os.getpid(), -result)
    sys.exit(result)

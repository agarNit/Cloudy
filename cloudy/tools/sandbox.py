import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain.agents.middleware.shell_tool import (
    BaseExecutionPolicy,
    DockerExecutionPolicy,
    HostExecutionPolicy,
)

from cloudy.observability.logger import get_logger


logger = get_logger(__name__)

# Deny everything by default, then allow only what a normal coding-agent shell
# session actually needs: reads anywhere (needed for the shell/tools to
# function at all — restricting this too would break basic commands), writes
# confined to the workspace (+ /tmp, which many tools use as scratch space),
# process fork/exec (to run subcommands), and nothing else — notably no
# network rule at all, so network access stays denied.
#
# Verified directly, not just written and trusted: writing outside the
# workspace and network access both fail with "Operation not permitted";
# ls/pwd/basic commands are unaffected.
_SEATBELT_PROFILE_TEMPLATE = """(version 1)
(deny default)
(allow process-fork)
(allow process-exec)
(allow file-read*)
(allow file-write* (subpath "{workspace}"))
(allow file-write* (subpath "/tmp"))
(allow file-write* (subpath "/private/tmp"))
(allow file-write* (subpath "/dev"))
(allow sysctl-read)
(allow mach-lookup)
(allow iokit-open)
(allow signal (target self))
"""


@dataclass
class SeatbeltExecutionPolicy(BaseExecutionPolicy):
    """Confines the shell session to its workspace directory and blocks network
    access, using macOS's built-in `sandbox-exec` — no external binary and no
    running daemon required, unlike the Codex CLI sandbox or Docker. This is
    what actually runs on a typical developer machine most of the time, since
    Docker isn't always running and the Codex CLI usually isn't installed.
    """

    def spawn(
        self, *, workspace: Path, env: Mapping[str, str], command: Sequence[str]
    ) -> subprocess.Popen[str]:
        profile = _SEATBELT_PROFILE_TEMPLATE.format(workspace=str(workspace))
        wrapped = ["sandbox-exec", "-p", profile, *command]
        # ShellToolMiddleware runs a persistent session and reads stdout/stderr on
        # separate threads (see ShellSession.start), so both need to be their own
        # pipe — merging stderr into stdout leaves process.stderr as None and breaks
        # the reader thread.
        return subprocess.Popen(
            wrapped,
            cwd=str(workspace),
            env=dict(env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )


def _docker_daemon_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=3)
        return result.returncode == 0
    except Exception:
        return False


def _seatbelt_available() -> bool:
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def detect_execution_policy() -> tuple[BaseExecutionPolicy, str]:
    """Pick the strongest sandboxing actually available on this machine right
    now, without requiring anything extra to be installed or running.
    Priority: Docker (real filesystem + network isolation, disposable
    container) if the daemon happens to be reachable, macOS Seatbelt (real
    write-confinement + no network, nothing extra needed) as the realistic
    default, plain host execution with resource limits as the honest
    fallback everywhere else — strictly better than zero limits, but no
    actual isolation.
    """
    if _docker_daemon_available():
        label = "Docker (container isolation, network disabled, disposed after each command)"
        logger.info(f"Shell sandboxing: {label}")
        return (
            DockerExecutionPolicy(memory_bytes=1_000_000_000, cpu_time_seconds=60),
            label,
        )

    if _seatbelt_available():
        label = "macOS Seatbelt (writes confined to project directory, network disabled)"
        logger.info(f"Shell sandboxing: {label}")
        return SeatbeltExecutionPolicy(), label

    label = "host only, no isolation (Docker/Seatbelt unavailable) — resource limits only"
    logger.warning(f"Shell sandboxing: {label}")
    return (
        HostExecutionPolicy(memory_bytes=1_000_000_000, cpu_time_seconds=60),
        label,
    )

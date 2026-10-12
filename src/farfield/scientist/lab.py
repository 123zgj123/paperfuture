"""Run a program somebody else wrote, and believe only what it printed.

An experiment is one Python file. It is started once per (condition, seed) with
a JSON argument file, and it reports by printing a single line::

    RESULT {"metric_name": 0.123, ...}

That line is the only thing the runtime reads a number from. A traceback, a
missing line or a timeout is a failed run and yields no number. Inside the line,
every finite number counts (nested objects are flattened to ``outer.inner``);
text, booleans and NaN are set aside with the reason and never become numbers.
A program that prints only those has produced no result.

The process runs in fresh user, mount, network and pid namespaces: no network,
the host's home and temp directories replaced by empty read-only mounts, the
root filesystem read-only, one writable working folder (also its ``TMPDIR``),
the lab's datasets read-only at the path in ``LAB_DATA``, a scrubbed environment
and hard CPU, memory and file-size limits. Every one of those steps must
succeed or the program does not start. Its output goes to files outside the
folder it can write, so it cannot rewrite what it printed. Programs written by a
model get to compute; they do not get the host.

The datasets are split once into an exploration pool and a confirmation pool.
Exploration and pre-freeze development see only the first; frozen confirmation
sees the second. This isolates phases, not all adaptive hypotheses: previously
reported confirmation results are still known to the research process.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import resource
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..services.parallel import map_parallel
from ..services import experiments

REPO = Path(__file__).resolve().parents[3]
RESULT_PREFIX = "RESULT "
TAIL = 2500
MEMORY_BYTES = 8 * 1024 ** 3          # the most one run may use
MIN_RUN_BYTES = 2 * 1024 ** 3         # below this, importing scipy and sklearn alone can fail
MEMORY_SHARE = 0.6                    # of what the machine has free, the lab may take
FILE_BYTES = 64 * 1024 ** 2
PACKAGES = ("numpy", "scipy", "sklearn", "networkx", "pandas", "statsmodels", "torch")

# Inside the namespace: stage the interpreter, the work folder and the datasets on a private
# tmpfs, blank out every place the host keeps personal files and scikit-learn's bundled
# datasets (they would be the same data in exploration and confirmation), make the rest
# read-only, and only then start the program. ``set -e``: a step that fails stops the launch.
_LAUNCH = r'''
set -e
stage="$1"; work="$2"; pyhome="$3"; python="$4"; data="$5"; files="$6"; selection="$7"; shift 7
mount -t tmpfs -o size=1m tmpfs "$stage"
mkdir "$stage/work"
mount --bind "$work" "$stage/work"
mkdir -p "$stage/work/.tmp"
if [ -n "$data" ]; then
  mkdir "$stage/data"
  if [ "$selection" = restricted ]; then
    mount --bind "$stage/data" "$stage/data"
    for name in $files; do
      touch "$stage/data/$name"
      mount --bind "$data/$name" "$stage/data/$name"
      mount -o remount,ro,bind "$stage/data/$name"
    done
    mount -o remount,ro,bind "$stage/data"
  else
    mount --bind "$data" "$stage/data"
    mount -o remount,ro,bind "$stage/data"
  fi
fi
if [ -n "$pyhome" ]; then
  mkdir "$stage/py"; mount --bind "$pyhome" "$stage/py"; python="$stage/py/bin/python"
  for bundled in "$stage"/py/lib*/python3*/site-packages/sklearn/datasets/data; do
    if [ -d "$bundled" ]; then mount -t tmpfs -o ro,size=4k tmpfs "$bundled"; fi
  done
fi
for hidden in "$@"; do
  if [ -d "$hidden" ]; then mount -t tmpfs -o ro,size=4k tmpfs "$hidden"; fi
done
# Every mount outside the stage becomes read-only — not only /, which leaves a network disk
# mounted under it writable — and the launch stops if any real filesystem is still writable.
awk '{print $2}' /proc/self/mounts | while read -r point; do
  case "$point" in "$stage"|"$stage"/*|/proc|/proc/*) continue ;; esac
  mount -o remount,ro,bind "$point" 2>/dev/null || true
done
# Only the topmost mount at a path is reachable; one shadowed beneath a hiding tmpfs is not.
writable=$(awk -v stage="$stage" '
  { top[$2] = $3 " " $4 }
  END { for (point in top) { split(top[point], f, " ")
    if (f[2] ~ /(^|,)rw(,|$)/ && point != stage && index(point, stage "/") != 1 && point !~ /^\/proc(\/|$)/ \
        && f[1] !~ /^(proc|sysfs|cgroup2?|devpts|devtmpfs|mqueue|debugfs|tracefs|securityfs|pstore|bpf|configfs|fusectl|hugetlbfs|efivarfs|binfmt_misc|autofs|rpc_pipefs|selinuxfs|nsfs)$/)
      print point } }' /proc/self/mounts)
if [ -n "$writable" ]; then echo "lab: still writable: $writable" >&2; exit 97; fi
cd "$stage/work"
exec "$python" -I experiment.py args.json
'''
_STAGES = ("/mnt", "/media", "/srv", "/opt")


@dataclass
class Run:
    ok: bool
    result: dict[str, float] = field(default_factory=dict)
    failure: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    seconds: float = 0.0
    returncode: int | None = None
    args: dict[str, Any] = field(default_factory=dict)
    set_aside: dict[str, str] = field(default_factory=dict)
    execution_receipt: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def python() -> Path:
    """The interpreter experiments run on: a lab environment if there is one."""
    chosen = os.environ.get("FARFIELD_LAB_PYTHON", "").strip()
    if chosen:
        return Path(chosen)
    local = REPO / "var" / "lab-venv" / "bin" / "python"
    return local if local.exists() else Path(sys.executable)


def data_dir() -> Path | None:
    """Real datasets the experiments may read (see scripts/fetch_lab_data.py), if any."""
    chosen = os.environ.get("FARFIELD_LAB_DATA", "").strip()
    folder = Path(chosen) if chosen else REPO / "var" / "lab-data"
    return folder if (folder / "manifest.json").is_file() else None


def datasets(pool: str = "") -> dict[str, Any]:
    """The catalogue, or one pool of it (``explore`` or ``confirm``)."""
    folder = data_dir()
    if folder is None:
        return {}
    try:
        found = dict(json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["datasets"])
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    if not pool:
        return found
    chosen = set(pools(found)[pool])
    return {name: row for name, row in found.items() if name in chosen}


EXPLORE_SHARE = 0.3


def pools(catalogue: Mapping[str, Any] | None = None) -> dict[str, list[str]]:
    """A fixed split of the catalogue: about 30% for looking, the rest for confirming.

    Fixed by the names alone, so every project and every rerun splits the same way, and no
    project can choose which datasets its exploration saw."""
    names = sorted(catalogue if catalogue is not None else datasets(),
                   key=lambda name: hashlib.sha256(f"farfield-pool:{name}".encode()).hexdigest())
    cut = int(len(names) * EXPLORE_SHARE)
    return {"explore": sorted(names[:cut]), "confirm": sorted(names[cut:])}


def pool_files(pool: str) -> list[str] | None:
    """The files a run in this pool may see, or None for the whole folder (no catalogue)."""
    folder = data_dir()
    if folder is None or not pool:
        return None
    names = set(datasets(pool))
    return sorted(path.name for path in folder.iterdir()
                  if path.is_file() and not path.is_symlink()
                  and path.name.split(".", 1)[0] in names)


def _venv_home(interpreter: Path) -> str:
    home = interpreter.parent.parent
    return str(home) if (home / "pyvenv.cfg").is_file() else ""


def _within(path: str, folder: str) -> bool:
    path, folder = os.path.abspath(path), os.path.abspath(folder)
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def _stage(*sources: Any) -> str:
    """Where the sandbox stages its mounts: an existing folder that is not above anything it
    must bind. A tmpfs over /mnt hides a work folder on /mnt/efs before it can be bound."""
    for candidate in _STAGES:
        if Path(candidate).is_dir() and not any(src and _within(str(src), candidate) for src in sources):
            return candidate
    return ""


_NETWORK_FS = ("nfs", "nfs4", "cifs", "smb3", "fuse.sshfs", "lustre", "ceph", "glusterfs")


def _network_mounts() -> list[str]:
    """Network filesystems on this host: shared disks hold other people's files."""
    try:
        rows = [line.split() for line in Path("/proc/self/mounts").read_text().splitlines()]
    except OSError:
        return []
    return [row[1] for row in rows if len(row) > 2 and row[2] in _NETWORK_FS]


def _hidden() -> list[str]:
    uid = os.getuid()
    places = [str(Path.home()), "/tmp", "/var/tmp", "/dev/shm", "/root", f"/run/user/{uid}",
              *_network_mounts()]
    for extra in (REPO,):
        if not any(str(extra).startswith(place.rstrip("/") + "/") for place in places):
            places.append(str(extra))
    return places


def _command(work: Path, interpreter: Path, stage: str, files: Sequence[str] | None,
             data_root: Path | None = None) -> list[str]:
    return ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork",
            "--mount-proc", "--propagation", "private",
            "sh", "-c", _LAUNCH, "lab", stage, str(work), _venv_home(interpreter),
            str(interpreter), str(data_root if data_root is not None else data_dir() or ""), " ".join(files or ()),
            "all" if files is None else "restricted", *_hidden()]


_PROBE = """
import json, os, socket
checks = {}
try:
    open("/farfield-probe", "w").write("x"); checks["root_read_only"] = 0.0
except OSError:
    checks["root_read_only"] = 1.0
checks["home_hidden"] = float(not os.path.isdir(HOME) or not os.listdir(HOME))
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2); checks["network_off"] = 0.0
except OSError:
    checks["network_off"] = 1.0
open(os.path.join(os.environ["TMPDIR"], "t"), "w").write("x")
open("w", "w").write("x")
checks["work_and_tmp_writable"] = 1.0
print("RESULT " + json.dumps(checks))
"""


@functools.lru_cache(maxsize=1)
def sandbox_available() -> bool:
    """Whether the whole launch works here and actually isolates: a probe program runs through
    the same path every experiment takes and must find the root read-only, the home folder
    empty, the network off, and its own folders writable."""
    if not shutil.which("unshare") or not _stage():
        return False
    folder = Path(tempfile.mkdtemp(prefix="farfield-probe-"))
    try:
        run = _launch(_PROBE.replace("HOME", repr(str(Path.home()))), {"seed": 0}, folder / "w",
                      timeout=60, interpreter=python(), memory=MIN_RUN_BYTES, files=None)
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return run.ok and all(value == 1.0 for value in run.result.values()) and len(run.result) == 4


def _available_bytes() -> int:
    """Memory the machine can give right now (MemAvailable), or a generous guess if unknown."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 4 * MEMORY_BYTES


def plan_resources(parallel: int) -> tuple[int, int]:
    """How many runs at once, and how much memory each, given what the machine has free now.

    A fixed 8 GB × 4 runs asks for more than many machines have; a live benchmark was killed
    for it. The lab takes at most MEMORY_SHARE of the free memory, shrinking first the memory
    per run and then the number of runs at once, never below one run of MIN_RUN_BYTES."""
    room = int(_available_bytes() * MEMORY_SHARE)
    parallel = max(1, min(int(parallel), os.cpu_count() or 1, room // MIN_RUN_BYTES or 1))
    per_run = max(MIN_RUN_BYTES, min(MEMORY_BYTES, room // parallel))
    return parallel, per_run


def _limits(timeout: float, memory: int = MEMORY_BYTES):
    def apply() -> None:
        cpu = int(timeout) + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def _parse(stdout: str) -> tuple[dict[str, float], str, dict[str, str]]:
    """(numbers, failure, set aside). Every finite number in the last RESULT line counts; the
    rest is set aside with the reason, so a program that also prints a label or a nested
    per-dataset table is not failed for it — only one that printed no number at all."""
    lines = [line for line in stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    if not lines:
        return {}, "no_result_line", {}
    try:
        payload = json.loads(lines[-1][len(RESULT_PREFIX):])
    except ValueError:
        return {}, "result_line_is_not_json", {}
    if not isinstance(payload, Mapping) or not payload:
        return {}, "result_is_not_an_object", {}
    numbers: dict[str, float] = {}
    aside: dict[str, str] = {}

    def walk(name: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, inner in value.items():
                walk(f"{name}.{key}" if name else str(key), inner)
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            aside[name] = f"not a number: {type(value).__name__}"
        elif not math.isfinite(value):
            aside[name] = f"not finite: {value}"
        else:
            numbers[name] = float(value)

    walk("", payload)
    if not numbers:
        return {}, "result_not_finite_numbers", aside
    return numbers, "", aside


def _tail(path: Path) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 4 * TAIL))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _result_line(path: Path) -> str:
    """The last RESULT line, read without loading an arbitrarily long output into memory."""
    last = ""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith(RESULT_PREFIX):
                    last = line.rstrip("\n")
    except OSError:
        pass
    return last


def _launch(code: str, args: Mapping[str, Any], work: Path, *, timeout: float, interpreter: Path,
            memory: int, files: Sequence[str] | None, data_root: Path | None = None) -> Run:
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / "experiment.py").write_text(code, encoding="utf-8")
    (work / "args.json").write_text(json.dumps(dict(args), sort_keys=True), encoding="utf-8")
    data = data_root if data_root is not None else data_dir()
    stage = _stage(work, data, _venv_home(interpreter))
    if not stage:
        return Run(ok=False, failure="no_stage_for_the_sandbox", args=dict(args))
    env = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LANG": "C.UTF-8",
           "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "TMPDIR": f"{stage}/work/.tmp", "JOBLIB_TEMP_FOLDER": f"{stage}/work/.tmp",
           "MPLCONFIGDIR": f"{stage}/work/.tmp", "LAB_DATA": f"{stage}/data" if data else ""}
    # The output lands beside the work folder, not in it: the program can write its folder,
    # and must not be able to rewrite what it printed. RLIMIT_FSIZE bounds both files.
    out_path, err_path = work.with_name(work.name + ".stdout"), work.with_name(work.name + ".stderr")
    started = time.monotonic()
    with open(out_path, "w") as out, open(err_path, "w") as err:
        process = subprocess.Popen(
            _command(work, interpreter, stage, files, data_root), stdout=out, stderr=err, text=True, env=env,
            cwd=str(work), start_new_session=True, preexec_fn=_limits(timeout, memory))
        try:
            process.wait(timeout=timeout)
            failure = ""
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            failure = "timeout"
    seconds = time.monotonic() - started
    result: dict[str, float] = {}
    aside: dict[str, str] = {}
    if not failure:
        if process.returncode != 0:
            failure = "crashed"
        else:
            result, failure, aside = _parse(_result_line(out_path))
    return Run(ok=not failure, result=result, failure=failure, stdout_tail=_tail(out_path)[-TAIL:],
               stderr_tail=_tail(err_path)[-TAIL:], seconds=round(seconds, 3),
               returncode=process.returncode, args=dict(args), set_aside=aside)


def run_once(code: str, args: Mapping[str, Any], *, workdir: Path, timeout: float = 60,
             interpreter: Path | None = None, memory: int | None = None, pool: str = "",
             data_root: Path | None = None, input_files: Sequence[str] | None = None) -> Run:
    """Execute ``code`` once. Refuses to run at all without the sandbox. ``pool`` limits the
    datasets the program can see (``explore`` or ``confirm``); empty shows the whole folder.
    A host-provided data_root and explicit input_files instead mount external raw
    assets. The model never supplies the host path or the exposure policy.
    """
    args = json.loads(experiments.canonical(dict(args)))
    chosen = Path(interpreter or python())
    budget = memory if memory is not None else plan_resources(1)[1]
    files = pool_files(pool)
    data = data_dir()
    input_digests = {}
    if data_root is not None:
        if pool or input_files is None:
            raise ValueError('external_assets_require_explicit_files_and_no_pool')
        data = Path(data_root).resolve(strict=True)
        files = list(input_files)
        for name in files:
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
                raise ValueError('invalid_external_asset_name')
            path = data / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('external_asset_must_be_a_regular_file')
            input_digests[name] = experiments.file_digest(path)
    elif input_files is not None:
        raise ValueError('external_files_require_data_root')
    spec = {"backend": "scientist.namespace_lab", "backend_digest": experiments.file_digest(Path(__file__)),
            "code_digest": hashlib.sha256(code.encode()).hexdigest(), "arguments": dict(args),
            "timeout_seconds": timeout, "memory_bytes": budget, "interpreter": str(chosen),
            "pool": pool, "files": files, "data_root": str(data) if data else "",
            "data_manifest_digest": experiments.file_digest(data / "manifest.json") if data and (data / 'manifest.json').is_file() else "",
            "input_digests": input_digests,
            "data_bytes_attested": False, "dependency_environment_attested": False}

    def launch() -> Run:
        if not sandbox_available():
            return Run(ok=False, failure="sandbox_unavailable", args=dict(args))
        options = {'data_root': data} if data_root is not None else {}
        result = _launch(code, args, Path(workdir), timeout=timeout, interpreter=chosen,
                         memory=budget, files=files, **options)
        if any(not (data / name).is_file() or experiments.file_digest(data / name) != digest
               for name, digest in input_digests.items()):
            result.ok, result.failure = False, 'input_assets_changed_during_execution'
        return result

    records = Path(workdir).with_name(Path(workdir).name + ".experiments")
    outcome = experiments.execute(spec, records=records, backend=launch,
                                  summarize=lambda run: run.to_dict(), succeeded=lambda run: run.ok)
    outcome.value.execution_receipt = {key: outcome.receipt[key] for key in
        ("attempt_id", "spec_digest", "result_digest", "status", "scientific_verdict")}
    outcome.value.execution_receipt["records"] = str(records.resolve())
    return outcome.value


def run_many(code: str, jobs: Iterable[Mapping[str, Any]], *, workdir: Path, timeout: float = 60,
             parallel: int = 4, pool: str = "") -> list[Run]:
    jobs = list(jobs)
    parallel, memory = plan_resources(parallel)

    def one(item: tuple[int, Mapping[str, Any]]) -> Run:
        index, args = item
        name = f"{index:03d}-{args.get('condition')}-{args.get('seed')}"
        if args.get("dataset"):
            name += f"-{args['dataset']}"
        return run_once(code, args, workdir=Path(workdir) / name, timeout=timeout, memory=memory, pool=pool)

    return map_parallel(one, enumerate(jobs), workers=parallel)


@functools.lru_cache(maxsize=1)
def _packages() -> dict[str, str]:
    probe = ("import importlib, json, sys\nfound = {}\n"
             f"for name in {PACKAGES!r}:\n"
             "    try:\n        found[name] = getattr(importlib.import_module(name), '__version__', 'yes')\n"
             "    except Exception:\n        pass\n"
             "print('PACKAGES ' + json.dumps(found))\n")
    try:
        out = subprocess.run([str(python()), "-I", "-c", probe], capture_output=True, text=True, timeout=120)
        for line in out.stdout.splitlines():
            if line.startswith("PACKAGES "):
                return json.loads(line[len("PACKAGES "):])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {}


def capabilities(pool: str = "") -> dict[str, Any]:
    """What an experiment can count on, as seen from one pool. Shown to whoever designs one."""
    shown = datasets(pool)
    where = ("the folder named by the environment variable LAB_DATA (read-only)" if shown
             else "none on disk: generate data")
    note = {"explore": "This is the exploration pool. Confirmation runs on the other datasets, never "
                       "on these, so what an exploration shows must hold on data it has not seen.",
            "confirm": "This is the confirmation pool. Pre-freeze development sees only the exploration "
                       "pool; registered dataset jobs are piloted on development datasets instead. "
                       "Confirmation values must not be used to repair the design. Previously observed "
                       "confirmation results are not a fresh independent holdout."}.get(pool, "")
    return {"python": "3.%d" % sys.version_info[1], "packages": _packages(),
            "standard_library": True, "network": False, "gpu": False,
            "cpu_cores": os.cpu_count() or 1, "memory_gb_per_run": round(plan_resources(4)[1] / 1024 ** 3, 1),
            "memory_note": "set from the memory the machine has free; keep one run well under it",
            "files": "only the working folder and TMPDIR are writable; the network is off",
            "datasets": {
                "where": where, "pool": pool or "all", "pool_note": note,
                "format": "np.load(os.path.join(os.environ['LAB_DATA'], '<name>.npz'), "
                          "allow_pickle=False) -> X (float64; categorical columns as integer codes; "
                          "missing values NaN), y (int class codes, or float for regression), "
                          "feature_names, categorical (bool mask), classes",
                "available": shown,
                "development_available": datasets("explore") if pool == "confirm" else shown,
                "not_available": "scikit-learn's bundled datasets (load_iris, load_digits, ...) are hidden: "
                                 "they would be the same data in exploration and confirmation"},
            "sandboxed": sandbox_available()}

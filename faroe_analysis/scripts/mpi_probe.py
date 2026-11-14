#!/usr/bin/env python3
"""Pinpoint WHY a Faroe fit is slow on a cluster. Runs in ~1 minute.

The fits do ~150 MPI allreduces per iteration (one per knot / parameter block), so a slow
collective transport is multiplied enormously: 200 ms per allreduce = ~30 s/iter. This
probe separates the three candidates:

  1. MPI not wired    -- every rank thinks it is rank 0 of a size-1 world (64 independent
                         serial copies, each doing ALL the times). Shows up as size=1.
  2. slow collectives -- size is right but Allreduce latency is milliseconds, not
                         microseconds (TCP fabric instead of shared memory).
  3. genuinely slow   -- collectives are fine and the per-iteration cost really is that
                         high, in which case the model/settings are the problem.

    mpiexec -n 64 python faroe_analysis/scripts/mpi_probe.py
"""
import os
import sys
import time
import warnings

warnings.simplefilter("ignore")                             # the radius-bump RuntimeWarning
                                                            # fires once per rank and floods
                                                            # the log, hiding [3]'s result.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                          # noqa: E402
from mpi4py import MPI                                      # noqa: E402

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# NB collectives must be called by EVERY rank -- never inside an `if rank == 0` block,
# or rank 0 blocks forever waiting for peers that skipped it.
hosts = comm.allgather(MPI.Get_processor_name())

if rank == 0:
    print("=" * 72, flush=True)
    print(f"[1] WORLD SIZE = {size}", flush=True)
    if size == 1:
        print("    *** BROKEN: size==1. The launcher started N INDEPENDENT serial jobs,",
              flush=True)
        print("    *** each computing ALL times. That alone explains a ~40x slowdown.",
              flush=True)
        print("    *** Cause is almost always an mpiexec/mpi4py MPI mismatch:", flush=True)
        print(f"    *** mpiexec on PATH : {os.popen('which mpiexec').read().strip()}",
              flush=True)
    print(f"    mpi4py built against: "
          f"{MPI.Get_library_version().strip().splitlines()[0]}", flush=True)
    print(f"    ranks report {len(set(hosts))} distinct hostname(s)", flush=True)

# --- [2] collective latency -------------------------------------------------
comm.Barrier()
for nbytes, label in ((8, "8 B (scalar, the common case)"), (8 * 432, "3.5 KB")):
    buf = np.ones(nbytes // 8, dtype=float)
    out = np.empty_like(buf)
    comm.Allreduce(buf, out, op=MPI.SUM)                    # warm
    comm.Barrier()
    t0 = time.time()
    N = 200
    for _ in range(N):
        comm.Allreduce(buf, out, op=MPI.SUM)
    dt = (time.time() - t0) / N
    dt = comm.allreduce(dt, op=MPI.MAX)
    if rank == 0:
        print(f"[2] Allreduce {label:32s}: {dt*1e6:9.1f} us/call", flush=True)
        if dt > 1e-3:
            print("    *** SLOW: >1 ms. Expect ~5-50 us on-node via shared memory.",
                  flush=True)
            print("    *** The fit does ~150 of these PER ITERATION -> "
                  f"~{150*dt:.1f} s/iter of pure communication.", flush=True)
            print("    *** Intel MPI: try  export I_MPI_FABRICS=shm", flush=True)
            print("    *** MPICH/OFI  : try  export FI_PROVIDER=shm", flush=True)
            print("    *** OpenMPI    : try  export OMPI_MCA_btl=self,vader", flush=True)

# --- [2b] per-core numerics micro-benchmark, SOLO vs ALL-RANKS-CONCURRENT ----
# Each time-slice costs BLAS (432x432 cholesky/solve) + RW C-ext (qRW). [2a] earlier
# runs of this were rank-0-only, i.e. the rank had the whole node to itself. The real fit
# runs all 64 ranks at once. If a solo cholesky is 2 ms but the SAME cholesky run
# concurrently on every rank is 300 ms, the primitives are fine and the killer is
# CONTENTION: 64 ranks each spawning BLAS threads onto the same cores (oversubscription),
# or unpinned ranks thrashing. That gap -- not the solo speed -- is what matches 105 s/iter.
import numpy as _np                                          # noqa: E402


def _bench(fn, n):
    fn()                                                    # warm
    t = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t) / n * 1e3                      # ms


Ns = 432
A = _np.random.default_rng(0).standard_normal((Ns, Ns))
SPD = A @ A.T + Ns * _np.eye(Ns)
b = _np.random.default_rng(1).standard_normal((Ns, 8))
try:
    import utilities as _u
    _phi = _np.full(Ns, 1.5); _gb = _np.full(Ns, 4.0); _p = _np.full(Ns, 0.95)
    _qrw = lambda: _u.qRW(_p, _phi, _gb, 2.0)               # noqa: E731
    _qrw()
except Exception:                                            # noqa: BLE001
    _qrw = None

# A memory-bandwidth streaming op on a working set that does NOT fit in cache. The
# cholesky/qRW micro-bench reuses one cached matrix, so it only sees ~5x contention. The
# real sampler streams Z/S/R (432x1656, ~6 MB each) per time-slice; 64 ranks hitting DRAM
# at once is a bandwidth wall the cache-resident bench cannot show. 96 MB/rank >> L3.
_big = _np.ones(12_000_000)                                  # 96 MB float64 per rank


def _stream():
    return float(_big.sum()) + float((_big * 1.0000001).sum())  # ~2 streaming reads


def _bwbench(n):
    _stream()
    t = time.time()
    for _ in range(n):
        _stream()
    dt = (time.time() - t) / n
    return _big.nbytes * 2 / dt / 1e9                        # GB/s (2 passes)


# transcendentals: exp/log dominate the RW density + Z-gradient but are ABSENT from
# cholesky/qRW/sum. If a numpy build lacks SIMD/SVML math, exp/log are ~10-50x slower
# while chol (BLAS) and sum stay identical -- which would explain [2b] matching across
# machines yet [3] differing 23x. This is the discriminator: numpy math vs MPI.
_xv = _np.random.default_rng(2).standard_normal(2_000_000)
_xp = _np.abs(_xv) + 0.5


def _trans():
    return _np.log(_np.exp(_xv * 0.5) + 1.0).sum() + _np.power(_xp, 0.37).sum()

# solo: only rank 0 works, everyone else waits -> the "whole node to myself" number
comm.Barrier()
if rank == 0:
    chol_solo = _bench(lambda: _np.linalg.cholesky(SPD), 50)
    qrw_solo = _bench(_qrw, 20) if _qrw else float("nan")
    bw_solo = _bwbench(10)
    trans_solo = _bench(_trans, 20)                          # 2M-elem exp+log+pow, ms
    trans_rate = 2000.0 / max(trans_solo, 1e-9)              # Melem/s (2e6 elem / ms)
comm.Barrier()
bw_conc = _bwbench(10)                                       # all ranks stream at once
bw_conc_min = comm.allreduce(bw_conc, op=MPI.MIN)            # worst rank's bandwidth
bw_conc_agg = comm.allreduce(bw_conc, op=MPI.SUM)            # total node bandwidth used
trans_c = _bench(_trans, 20)                                 # exp/log/pow, all ranks
trans_cmax = comm.allreduce(trans_c, op=MPI.MAX)

# concurrent: EVERY rank runs the same bench at the same time -> the real-fit condition
chol_c = _bench(lambda: _np.linalg.cholesky(SPD), 50)
qrw_c = _bench(_qrw, 20) if _qrw else float("nan")
chol_cmax = comm.allreduce(chol_c, op=MPI.MAX)
qrw_cmax = comm.allreduce(qrw_c, op=MPI.MAX)
naff = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else -1
naff_max = comm.allreduce(naff, op=MPI.MAX)
thr = os.environ.get("OMP_NUM_THREADS", "<unset>")
mkl = os.environ.get("MKL_NUM_THREADS", "<unset>")
obl = os.environ.get("OPENBLAS_NUM_THREADS", "<unset>")


def _numa_info():
    # NUMA nodes that PHYSICALLY exist on the box, vs the ones THIS rank may allocate on
    # (Mems_allowed_list -- Slurm can confine a job's memory to one node even with all
    # cores). If allowed==1 while the box has several, `numactl --interleave=all` is a
    # no-op and the job is capped at one domain's bandwidth -- a Slurm --mem-bind issue.
    ndev = "?"
    try:
        ndev = sum(1 for n in os.listdir("/sys/devices/system/node")
                   if n.startswith("node") and n[4:].isdigit())
    except Exception:
        pass
    allowed = "?"
    try:
        with open("/proc/self/status") as fh:
            for ln in fh:
                if ln.startswith("Mems_allowed_list:"):
                    allowed = ln.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return ndev, allowed


_nnode, _allowed = _numa_info()

if rank == 0:
    print(f"[2b] cholesky(432)  SOLO {chol_solo:7.2f} ms   "
          f"ALL-{size}-CONCURRENT {chol_cmax:7.2f} ms   "
          f"({chol_cmax/max(chol_solo,1e-9):.0f}x)", flush=True)
    if not (_qrw is None):
        print(f"     qRW(432)       SOLO {qrw_solo:7.2f} ms   "
              f"ALL-{size}-CONCURRENT {qrw_cmax:7.2f} ms   "
              f"({qrw_cmax/max(qrw_solo,1e-9):.0f}x)", flush=True)
    print(f"     exp/log/pow (2M-elem)  SOLO {trans_solo:7.2f} ms ({trans_rate:6.1f} "
          f"Melem/s)   ALL-{size} {trans_cmax:7.2f} ms   [numpy MATH -- absent from "
          f"chol/qRW]", flush=True)
    print(f"        >>> COMPARE THIS across machines: chol/qRW/bandwidth match Alpine<->"
          f"the reference workstation, so if exp/log differs a lot HERE, slow numpy math is the 23x.",
          flush=True)
    bw_slow = bw_solo / max(bw_conc_min, 1e-9)
    print(f"     mem-bandwidth (96MB stream)  SOLO {bw_solo:6.1f} GB/s   "
          f"per-rank when ALL-{size} {bw_conc_min:6.2f} GB/s   ({bw_slow:.0f}x slower)",
          flush=True)
    print(f"        node aggregate under load: {bw_conc_agg:6.1f} GB/s "
          f"(saturated bus -> this is the DRAM ceiling)", flush=True)
    print(f"     per-rank CPUs in affinity mask: {naff_max}  "
          f"(want 1 each; {size} means every rank floats over all cores)", flush=True)
    print(f"     thread caps seen by ranks: OMP={thr} MKL={mkl} OPENBLAS={obl}",
          flush=True)
    print(f"     NUMA: box has {_nnode} node(s); this job may allocate on nodes "
          f"[{_allowed}]", flush=True)
    if str(_nnode).isdigit() and int(_nnode) > 1 and _allowed in ("0", "1", "2", "3"):
        print(f"     *** MEMORY CONFINED to 1 of {_nnode} NUMA nodes -- so all {size} "
              "ranks share ONE domain's channels (~the 55 GB/s ceiling) and "
              "numactl --interleave=all CANNOT spread (nothing to spread onto).",
              flush=True)
        print("     *** This is a Slurm allocation setting, not the code. Try adding to "
              "the sbatch:", flush=True)
        print("     ***   #SBATCH --mem-bind=none      (don't confine memory)", flush=True)
        print("     *** or request the whole node so all NUMA memory is yours:", flush=True)
        print("     ***   #SBATCH --exclusive          (and/or --ntasks-per-node with "
              "all cores)", flush=True)
        print("     *** If it persists, this is one for rc-help@colorado.edu: 64-core "
              "AMD, 25x slower than an equivalent workstation, memory pinned to one "
              "NUMA node.", flush=True)
    if bw_slow > 10:
        print(f"     *** MEMORY-BANDWIDTH BOUND: each rank's streaming bandwidth "
              f"collapses {bw_slow:.0f}x under {size}-way load. The 432x432 chol only "
              "showed 5x because it fits in cache; the real sampler streams Z/S/R and "
              "hits this wall.", flush=True)
        print("     *** Fixes, cheapest first: (1) run FEWER ranks -- a bandwidth-bound "
              "job is often FASTER at 16-32 ranks than 64 (less bus contention, each rank "
              "faster). (2) bind ranks NUMA-local (--bind-to core --map-by numa) so memory "
              "is local, not cross-socket.", flush=True)
    if chol_cmax > 4 * max(chol_solo, 1e-9):
        print(f"     *** CONTENTION: concurrent cholesky is {chol_cmax/chol_solo:.0f}x "
              "the solo cost. The primitives are FINE; the ranks are fighting for cores.",
              flush=True)
        print("     *** Fix = one core per rank. In _faroe_env.sh / the sbatch, add:",
              flush=True)
        print("     ***   export OMPI_MCA_hwloc_base_binding_policy=core   (OpenMPI)",
              flush=True)
        print("     ***   and launch with: mpiexec --bind-to core -n $NRANKS ...",
              flush=True)
        print("     *** If thread caps above are NOT 1, also export them =1 (a threaded",
              flush=True)
        print("     *** BLAS with 64 ranks x 64 threads is the classic oversubscription).",
              flush=True)
    blas = ""
    try:                                                     # numpy >=1.25 returns a dict
        cfg_ = _np.__config__.show(mode="dicts")
        blas = str(cfg_.get("Build Dependencies", {}).get("blas", {}).get("name", ""))
    except Exception:
        try:                                                 # older numpy prints to stdout
            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _np.show_config()
            for ln in buf.getvalue().splitlines():
                if any(k in ln.lower() for k in ("mkl", "openblas", "blas", "lapack")):
                    blas += ln.strip() + " | "
        except Exception:
            pass
    print(f"     numpy BLAS: {blas or '<unknown; run numpy.show_config()>'}", flush=True)
    # threadpoolctl: the ACTUAL native math libraries loaded + their real thread counts
    # (env vars only report what we ASKED for). Compare this line across machines -- a
    # different BLAS build or a second OpenMP runtime would leave chol identical yet make
    # the sampler's heavier numpy paths diverge.
    try:
        import threadpoolctl
        pools = threadpoolctl.threadpool_info()
        if not pools:
            print("     threadpoolctl: no native pools detected", flush=True)
        for p in pools:
            print(f"     threadpool: {p.get('user_api','?')}/{p.get('internal_api','?')} "
                  f"{os.path.basename(str(p.get('filepath','?')))} "
                  f"v{p.get('version','?')}  num_threads={p.get('num_threads','?')}",
                  flush=True)
    except ImportError:
        print("     threadpoolctl: not installed (pip install threadpoolctl) -- it would "
              "show the REAL BLAS/OpenMP libs + thread counts; compare across machines.",
              flush=True)
    except Exception as _e:                                   # noqa: BLE001
        print(f"     threadpoolctl: {_e}", flush=True)

# --- [2d] interleaved compute+allreduce: the REAL MCMC-iteration shape -------
# [2] times bare allreduces (all ranks pre-Barrier'd, no compute between them) and [2b]
# times compute with NO collectives. The fit does NEITHER in isolation: one MCMC
# iteration is ~150 MH blocks, and each block is (this rank computes its share of the
# per-time log-likelihood) THEN (a global Allreduce to sum it). With unequal per-rank
# work -- different time counts, deeper qRW root-finds on some ranks -- every allreduce
# becomes a straggler barrier, and on UNPINNED ranks (mask=64 above) the fast ones
# busy-wait-spin and steal cores from the straggler still computing. That interaction is
# invisible to [2]/[2b] yet is exactly what one MCMC iteration pays -- the leading
# suspect for a 23x that no per-primitive number (chol/qRW/exp/log/bandwidth all match
# across machines) can explain. This reproduces it in ~1 s.
NBLK = int(os.environ.get("PROBE_NBLK", "150"))               # ~allreduces per MCMC iter
_nwork = 1 + (1 if rank >= size // 2 else 0)                  # upper half = 2x stragglers
_s1 = _np.ones(1)
_s1o = _np.empty(1)


def _localwork():                                             # a rank's per-block BLAS
    for _ in range(_nwork):
        _np.linalg.cholesky(SPD)


_localwork()                                                  # warm
comm.Barrier()
_t = time.time()
for _ in range(NBLK):                                         # compute only, no sync
    _localwork()
_comp = comm.allreduce(time.time() - _t, op=MPI.MAX)
comm.Barrier()
_t = time.time()
for _ in range(NBLK):                                         # bare allreduces, no compute
    comm.Allreduce(_s1, _s1o, op=MPI.SUM)
_bare = comm.allreduce(time.time() - _t, op=MPI.MAX)
comm.Barrier()
_t = time.time()
for _ in range(NBLK):                                         # the real shape: work + sync
    _localwork()
    comm.Allreduce(_s1, _s1o, op=MPI.SUM)
_inter = comm.allreduce(time.time() - _t, op=MPI.MAX)
if rank == 0:
    _ideal = _comp + _bare                                    # if sync overlapped for free
    _ratio = _inter / max(_ideal, 1e-9)
    print(f"[2d] {NBLK} x (imbalanced cholesky + 1 Allreduce) -- one MCMC iteration's "
          f"shape", flush=True)
    print(f"     compute-only {_comp*1e3:7.1f} ms   bare-allreduce {_bare*1e3:7.1f} ms   "
          f"ideal-sum {_ideal*1e3:7.1f} ms", flush=True)
    print(f"     INTERLEAVED  {_inter*1e3:7.1f} ms   => {_ratio:4.1f}x ideal "
          f"(sync / straggler-barrier / busy-wait-steal overhead)", flush=True)
    if _ratio > 2.0:
        print(f"     *** SYNC-BOUND: interleaving compute with {NBLK} collectives costs "
              f"{_ratio:.0f}x more than doing the compute and the comm separately. This is "
              "the regime the fit runs in and [2]/[2b] cannot see. Fixes: pin ranks "
              "(sbatch --export=ALL,FAROE_MPI_BIND=core; mask should drop 64->1 above), and "
              "on OpenMPI export OMPI_MCA_mpi_yield_when_idle=1 so idle ranks yield instead "
              "of spinning. Re-run pinned and watch this ratio + [3] s/iter fall together.",
              flush=True)
    else:
        print("     (interleaving is ~free here -- collectives overlap cleanly, so the 23x "
              "is NOT the sync path; look at [2b] concurrency, [2c] I/O, or real per-iter "
              "work.)", flush=True)

# --- [2c] STORAGE write bandwidth: latent target vs node-local --------------
# The real fit STREAMS ~43 GB of Z + 4 GB of S latents to a memmap and checkpoints every
# 50 iters -- none of which the compute-only [3] does. If FAROE_OUT is on /home (NFS,
# small quota) or /projects (GPFS) instead of /scratch/alpine (Lustre), those writes stall.
#
# HANG-SAFE: rank 0 only (no collective that could deadlock if one rank's write is slow),
# with a SIGALRM hard timeout. The latent stream is rank-0-anyway (gather then rank 0
# writes the memmap), so this is the relevant path. A TIMEOUT here IS the verdict: the
# filesystem cannot absorb the write in N seconds.
if rank == 0:
    import signal

    class _Timeout(Exception):
        pass

    def _write_bw(path_dir, mb, timeout_s=25):
        def _alarm(sig, frame):
            raise _Timeout()
        os.makedirs(path_dir, exist_ok=True)
        buf = _np.ones(mb * 131072, dtype=_np.float64).tobytes()   # mb megabytes
        fp = os.path.join(path_dir, ".mpiprobe_io.bin")
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout_s)
        try:
            f = open(fp, "wb", buffering=0)                        # unbuffered: the FS
            t = time.time()
            f.write(buf); f.flush(); os.fsync(f.fileno())          # force to storage
            dt = time.time() - t
            f.close(); os.remove(fp)
            return f"{mb/dt:7.1f} MB/s", None
        except _Timeout:
            try:
                os.remove(fp)
            except OSError:
                pass
            return f">{timeout_s}s for {mb}MB (<{mb/timeout_s:.1f} MB/s)", "TIMEOUT"
        except Exception as e:                                     # noqa: BLE001
            return "n/a", f"{type(e).__name__}: {e}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    def _fsname(p):
        try:
            import subprocess
            return subprocess.run(["stat", "-f", "-c", "%T", p], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or "?"
        except Exception:
            return "?"

    _latent_dir = os.environ.get("FAROE_OUT",
                                 os.path.join(os.getcwd(), "faroe_analysis", "output"))
    _local_dir = os.environ.get("FAROE_OMPI_TMPDIR", "/dev/shm")
    print(f"[2c] STORAGE write (rank 0, 128MB, fsync'd; the latent-stream path) ...",
          flush=True)
    print(f"     latent target {_latent_dir} [{_fsname(_latent_dir)}]: ", end="", flush=True)
    lat, lat_err = _write_bw(_latent_dir, 128)
    print(lat + (f"   ERR {lat_err}" if lat_err else ""), flush=True)
    print(f"     node-local    {_local_dir} [{_fsname(_local_dir)}]: ", end="", flush=True)
    loc, loc_err = _write_bw(_local_dir, 128)
    print(loc + (f"   ERR {loc_err}" if loc_err else ""), flush=True)
    if lat_err == "TIMEOUT" or "MB/s" in lat:
        print("     NB the real fit ALSO checkpoints all 64 ranks at once every 50 iters,",
              flush=True)
        print("        so contended write pressure is worse than this rank-0 number.",
              flush=True)
    if lat_err == "TIMEOUT" or (lat_err is None and loc_err is None and "MB/s" in lat
                                and float(lat.split()[0]) < 50):
        print("     *** SLOW LATENT STORAGE. Put FAROE_OUT on fast scratch and check less "
              "often:", flush=True)
        print("     ***   export FAROE_OUT=/scratch/alpine/$USER/faroe_out   (Lustre)",
              flush=True)
        print("     ***   export FAROE_CKPT_HOURS=2   (avoid the every-50-iters floor)",
              flush=True)
comm.Barrier()                                               # rejoin before [3]

# --- [3] one real sampler iteration ----------------------------------------
if os.environ.get("PROBE_SKIP_SAMPLER"):
    if rank == 0:
        print("[3] skipped (PROBE_SKIP_SAMPLER set); [2b] is the per-core verdict.",
              flush=True)
        print("=" * 72, flush=True)
    sys.exit(0)
if rank == 0:
    print("[3] timing real sampler iterations (slow at 64 ranks; set PROBE_SKIP_SAMPLER "
          "to skip) ...", flush=True)
try:
    import faroe_setup as F
    from sampler_mpi import run_mcmc_mpi
    from sampler import SamplerSettings
    inp = F.load_inputs()
    cfg, kw = F.models()["wang"]
    data = F.build_data(cfg, inp, "full")
    base = {**F.base_settings_kw(), **kw, "save_latent": False, "stream_latent": False,
            "latent_out": None, "plot_every": 0, "checkpoint_every": 0, "resume": False,
            "checkpoint_path": None, "thin": 1}
    r = os.environ.get("PROBE_RADIUS")
    if r:
        base["radius_S"] = float(r); base["update_radius_S"] = False
    ntimes = data.Y.shape[1]

    # One quick timing with the default (nuts) sampler. The point is to compare ACROSS
    # rank counts: one iteration processes all `ntimes` regardless of how many ranks share
    # them, so s/iter should FALL ~linearly with more ranks IF the node has the resources.
    # If s/iter is flat from 16->64 ranks, the node is throughput-saturated (bandwidth) and
    # more ranks buy nothing -- run the fits at the knee. Re-run this probe with NRANKS=16,
    # 32, 64 (sbatch --export=...,NRANKS=32) and read the "times/sec" column.
    skw = {**base, "z_sampler": "nuts", "z_update": None,
           "nuts_warmup": F.NUTS_WARMUP, "nuts_target_accept": F.NUTS_TARGET_ACCEPT}
    NIT = int(os.environ.get("PROBE_ITERS", "3"))
    run_mcmc_mpi(cfg, data, SamplerSettings(**{**skw, "n_iters": 2}))       # warm
    comm.Barrier()
    t0 = time.time()
    run_mcmc_mpi(cfg, data, SamplerSettings(**{**skw, "n_iters": NIT}))
    my = time.time() - t0
    per = comm.allreduce(my, op=MPI.MAX) / NIT             # slowest rank sets the pace
    if rank == 0:
        tps = ntimes / per                                 # time-slices processed per sec
        print(f"    {size:3d} ranks: {per:7.2f} s/iter   "
              f"{tps:7.1f} time-slices/sec (node throughput)   "
              f"=> 75k iters = {per*75000/3600:5.0f} h", flush=True)
        print(f"    Reference workstation (Wang, 64 ranks): ~4.3 s/iter. Re-run at NRANKS=16 "
              f"and 32:", flush=True)
        print(f"      if time-slices/sec is FLAT vs rank count -> node-saturated, use the "
              f"knee; if it SCALES -> more ranks still help.", flush=True)
except Exception as e:                                       # noqa: BLE001
    if rank == 0:
        print(f"    (sampler timing skipped: {e})", flush=True)
if rank == 0:
    print("=" * 72, flush=True)

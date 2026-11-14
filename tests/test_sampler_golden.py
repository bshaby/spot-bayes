"""Golden-regression gate for the serial and MPI samplers.

Both drivers are deterministic run-to-run (fixed seeds, 1 thread). The committed
fixtures in fixtures/sampler_golden/ were captured from the sampler before the
serial/MPI update-block unification; these tests require that the live samplers
still reproduce them byte-for-byte. This is what lets us refactor the duplicated
update blocks with confidence: behaviour must not change.

There are TWO fixtures per backend. serial/mpi2 run the RWMH samplers; serial_nuts/
mpi2_nuts run z_sampler="nuts" + nu_rho_sampler="nuts", which is the only thing that
exercises the gradient blocks -- zblock_value_grad_* and _nu_rho_cp_value_grad*. The
RWMH pair does not touch them at all, so before the _nuts pair existed, changing the
default gradient implementation for either block moved no golden.

If a *deliberate* numerical change is made, regenerate the fixtures with:

    P=$(python -c "import sys; print(sys.executable)")
    M=$(dirname $P)/mpiexec          # NOT a bare `mpiexec` -- see below
    $P tests/_sampler_golden_runner.py tests/fixtures/sampler_golden/serial.npz serial
    $P tests/_sampler_golden_runner.py tests/fixtures/sampler_golden/serial_nuts.npz serial_nuts
    $M -n 2 $P tests/_sampler_golden_runner.py tests/fixtures/sampler_golden/mpi2.npz mpi
    $M -n 2 $P tests/_sampler_golden_runner.py tests/fixtures/sampler_golden/mpi2_nuts.npz mpi_nuts

The MPI launcher MUST be the one beside this interpreter, which is what
_find_mpi_launcher picks and therefore what the test compares against. A bare
`mpiexec` may resolve to a system MPI (/usr/bin/mpiexec) that cannot launch an
interpreter whose mpi4py links a different MPI: the ranks then run as independent
size-1 jobs, the run SUCCEEDS, and the fixture it writes is not a 2-rank result at
all. Regenerating that way produces goldens that fail against every correct run --
observed, and it looks like sampler drift rather than a launcher mismatch.
"""
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "_sampler_golden_runner.py")
GOLDEN = os.path.join(HERE, "fixtures", "sampler_golden")


def _find_mpi_launcher():
    """Locate an MPI launcher that matches the running interpreter's mpi4py.

    A bare PATH lookup is unsafe: a system mpirun (e.g. Open MPI) won't
    correctly launch an interpreter whose mpi4py is linked against a different
    MPI (e.g. Intel MPI / MPICH) -- the ranks then run as independent size-1
    jobs and the golden comparison is meaningless. So prefer a launcher that
    lives next to sys.executable (same env), falling back to PATH only as a
    last resort. Returns (launcher_path, bindir) or (None, None).
    """
    bindir = os.path.dirname(os.path.abspath(sys.executable))
    for name in ("mpiexec", "mpirun"):
        cand = os.path.join(bindir, name)
        if os.path.exists(cand):
            return cand, bindir
    from shutil import which
    for name in ("mpiexec", "mpirun"):
        cand = which(name)
        if cand:
            return cand, os.path.dirname(cand)
    return None, None


def _assert_matches_golden(live, golden_path):
    g = np.load(golden_path)
    assert set(live.files if hasattr(live, "files") else live.keys()) == set(g.files)
    assert len(g.files) > 0
    for k in g.files:
        lv = live[k]
        assert np.array_equal(lv, g[k]), f"sampler output '{k}' drifted from golden"


@pytest.mark.parametrize("mode,fixture", [("serial", "serial.npz"),
                                          ("serial_nuts", "serial_nuts.npz")])
def test_serial_matches_golden(mode, fixture):
    sys.path.insert(0, os.path.dirname(HERE))
    import _sampler_golden_runner as r
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TR, _ = r.run(mode)
    live = {k: np.asarray(TR[k]) for k in r.GLOBAL_KEYS if k in TR}
    _assert_matches_golden(_DictFiles(live), os.path.join(GOLDEN, fixture))


@pytest.mark.parametrize("mode,fixture", [("mpi", "mpi2.npz"),
                                          ("mpi_nuts", "mpi2_nuts.npz")])
def test_mpi_matches_golden(tmp_path, mode, fixture):
    launcher, bindir = _find_mpi_launcher()
    if launcher is None:
        pytest.skip("no MPI launcher (mpiexec/mpirun) available")
    out = str(tmp_path / fixture)
    env = dict(os.environ)
    # ensure the env's own MPI hydra helpers (mpiexec.hydra etc.) are found
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1")
    try:
        subprocess.run([launcher, "-n", "2", sys.executable, RUNNER, out, mode],
                       check=True, env=env, cwd=HERE, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, timeout=600)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"MPI child failed: {e.stderr.decode()[-2000:]}")
    if not os.path.exists(out):
        pytest.skip(f"{launcher} did not produce rank-0 output (incompatible "
                    "launcher/mpi4py pairing); ran as independent processes")
    _assert_matches_golden(np.load(out), os.path.join(GOLDEN, fixture))


_MPI_PLOT_CHILD = '''
import os, sys
import simulate as sm
from config import ModelConfig
from sampler import SamplerData, SamplerSettings
from sampler_mpi import run_mcmc_mpi
from mpi4py import MPI
png = sys.argv[1]
cfg = ModelConfig(model_type="shi")
out = sm.simulate(cfg, sm.SimConfig(Ns=20, Nt=10, seed=5))
data = SamplerData(Y=out["Y_missing"], sites_xy=out["sites_xy"],
    u_vec=out["u_mat"][:, 0], sigma_vec=out["sigma_mat"][:, 0],
    xi_vec=out["xi_mat"][:, 0], p_cens=out["p_cens"], knots_S=out["knots_S"],
    knots_phi=out["knots_phi"], knots_rho=out["knots_rho"])
# thin=10, plot_every=30: the old MPI trigger ((it+1)%plot_every) could never
# coincide with the recording grid (it%thin==0), so no plot was ever written.
st = SamplerSettings(n_iters=40, thin=10, adapt_size=20, nu=out["nu"], seed=1,
                     plot_every=30, plot_path=png)
run_mcmc_mpi(cfg, data, st)
'''


def test_mpi_writes_live_plot(tmp_path):
    """Regression: the MPI live-plot trigger must fire on the recording grid.

    Previously the plot block was nested inside `if it % thin == 0` but used the
    `(it+1) % plot_every` phase, so for any exp1-style plot_every (a multiple of
    thin) the two conditions never coincided and NO live plot was ever written
    under MPI -- silently (no exception). This launches a tiny 2-rank run with
    that exact alignment and asserts the PNG appears."""
    launcher, bindir = _find_mpi_launcher()
    if launcher is None:
        pytest.skip("no MPI launcher (mpiexec/mpirun) available")
    child = tmp_path / "_mpi_plot_child.py"
    child.write_text(_MPI_PLOT_CHILD)
    png = str(tmp_path / "live.trace.png")
    env = dict(os.environ)
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.path.dirname(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1")
    try:
        subprocess.run([launcher, "-n", "2", sys.executable, str(child), png],
                       check=True, env=env, cwd=HERE, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, timeout=600)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"MPI child failed: {e.stderr.decode()[-2000:]}")
    assert os.path.exists(png) and os.path.getsize(png) > 1000


class _DictFiles(dict):
    """Give a plain dict the np.load-like .files attribute used above."""
    @property
    def files(self):
        return list(self.keys())

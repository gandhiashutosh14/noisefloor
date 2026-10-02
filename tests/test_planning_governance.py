import json

import numpy as np
import pytest

from noisefloor.agents.rules import HPA, Context
from noisefloor.agents.wm_agent import rollout_actuator
from noisefloor.audit.envelopes import HAVE_STERNWATCH, EnvelopeRecorder, envelope_dict, run_id
from noisefloor.config import sim_config
from noisefloor.eval.stats import hierarchical_bootstrap, holm, iqm, seed_t_interval, tost
from noisefloor.govern.gate import Gate
from noisefloor.plan.cem import cem_plan
from noisefloor.sim.fleet import (
    ACTIONS,
    FAILOVER,
    NOOP,
    SHED,
    UP2,
    UP6,
    advance_time,
    allowed_mask,
    apply_actions,
    new_actuator,
)

CFG = sim_config()


# ------------------------------------------------------------------------------------------ CEM
def test_cem_finds_a_hidden_target_sequence():
    target = np.array([3, 1, 4, 1, 5, 2, 6, 0])

    def cost(seqs):
        return (seqs != target[None, None, :]).sum(-1).astype(float)

    ranked, spread, p = cem_plan(2, 7, cost, np.random.default_rng(0), horizon=8, samples=256, elites=16, iterations=6)
    assert (p.argmax(-1) == target).all()
    assert (ranked[:, 0] == target[0]).all()


def test_cem_ranks_on_the_effective_first_action():
    """A refused first action is scored as a no-op inside the rollouts, so it must not be ranked as
    itself: with the mask it is folded into the no-op plan and never handed to the gate."""
    mask = np.ones((3, 7), dtype=bool)
    mask[:, [3, 4]] = False
    flat = lambda seqs: np.zeros(seqs.shape[:2])                       # every plan costs the same
    ranked, _, _ = cem_plan(3, 7, flat, np.random.default_rng(0), horizon=4, samples=64, first_mask=mask)
    assert not np.isin(ranked, [3, 4]).any()
    assert (mask[np.arange(3)[:, None], ranked]).all()


def test_gate_limits_agree_with_the_mask_for_a_stricter_catalog():
    """gate_v2 caps replicas at 24 and has no failover: the mask built from its limits and the gate's
    own verdicts must agree on every reachable state (v1 is covered by the exhaustive test above)."""
    gate = Gate.load("gate_v2.json")
    limits = gate.limits(CFG)
    assert limits["replicas_max"] == 24 and "failover" not in limits["allowed_actions"]
    rng = np.random.default_rng(2)
    act = new_actuator(rng.integers(2, 25, 16), CFG, budget_window=gate.window)
    for _ in range(40):
        m = allowed_mask(act, CFG, limits)
        for i in range(16):
            for a, name in enumerate(ACTIONS):
                assert (gate.check(name, act, i, CFG, approved=True).verdict == "allow") == bool(m[i, a]), (i, name)
        choice = np.array([rng.choice(np.flatnonzero(row)) for row in m])
        apply_actions(act, choice, CFG)
        advance_time(act)
    assert (act.r + act.pending <= 24).all()


def test_cem_rollouts_respect_the_gate_mask():
    act = new_actuator(np.array([39]), CFG)                     # scale-ups over 40 are refused
    U, A, _ = rollout_actuator(act, np.full((1, 4), UP6), CFG)
    assert (A == NOOP).all()
    U, A, _ = rollout_actuator(new_actuator(np.array([10]), CFG), np.full((1, 8), UP2), CFG)
    assert (A[0, :6] == UP2).all() and (A[0, 6:] == NOOP).all()  # budget: 6 actions per 12 steps


def test_rollout_delays_failover_by_one_step():
    U, A, _ = rollout_actuator(new_actuator(np.array([10]), CFG), np.array([[FAILOVER, NOOP, NOOP, FAILOVER]]), CFG)
    assert list(A[0]) == [NOOP, FAILOVER, NOOP, NOOP]           # executed once, then locked out


# ------------------------------------------------------------------------------------------ gate
def test_gate_agrees_with_the_vectorised_mask():
    gate = Gate.load("gate_v1.json")
    rng = np.random.default_rng(1)
    act = new_actuator(rng.integers(2, 41, 32), CFG)
    for _ in range(40):
        m = allowed_mask(act, CFG)
        for i in range(32):
            for a, name in enumerate(ACTIONS):
                v = gate.check(name, act, i, CFG, approved=True)
                assert (v.verdict == "allow") == bool(m[i, a]), (i, name, v)
        choice = np.array([rng.choice(np.flatnonzero(row)) for row in m])
        apply_actions(act, choice, CFG)
        advance_time(act)


def test_gate_catalog_rejects_irreversible_without_approval():
    from noisefloor.config import CONFIGS
    cat = json.loads((CONFIGS / "gate_v1.json").read_text(encoding="utf-8"))
    for c in cat["capabilities"]:
        if c["name"] == "failover":
            c["requires_approval"] = False
    with pytest.raises(ValueError, match="irreversible"):
        Gate(cat)


def test_gate_verdicts_budget_and_compensation():
    gate = Gate.load("gate_v1.json")
    act = new_actuator(np.array([10]), CFG)
    assert gate.check("failover", act, 0, CFG).verdict == "needs-approval"
    assert gate.check("shed_10", act, 0, CFG).effect == "compensable"
    assert gate.caps["shed_10"]["compensation"] == "restore_traffic"
    for _ in range(6):
        apply_actions(act, np.array([UP2]), CFG)
    v = gate.check("scale_up_2", act, 0, CFG)
    assert v.verdict == "deny" and "budget" in v.reason
    act2 = new_actuator(np.array([2]), CFG)
    v = gate.check("scale_down_2", act2, 0, CFG)
    assert v.verdict == "deny" and "minimum" in v.reason


def test_shed_is_compensated_after_the_restore_window():
    act = new_actuator(np.array([10]), CFG)
    apply_actions(act, np.array([SHED]), CFG)
    assert act.shed[0] == pytest.approx(0.1)
    for _ in range(CFG["shed_restore_steps"]):
        advance_time(act)
    assert act.shed[0] == pytest.approx(0.0)


# ------------------------------------------------------------------------------------------ HPA
def _ctx(r, rho, pending=0):
    act = new_actuator(np.array([r]), CFG)
    act.p1 = np.array([pending])
    return Context(t=0, obs=None, golden={"rho": np.array([rho]), "error_rate": np.zeros(1), "load": np.zeros(1),
                                          "p95_ms": np.zeros(1)}, actuator=act, act_vec=None, mask=None)


@pytest.mark.parametrize("r,rho,expected", [
    (10, 0.60, 10),     # on target
    (10, 0.65, 10),     # within the 10% tolerance
    (10, 0.90, 15),     # ceil(10 * 0.9 / 0.6) = 15
    (10, 0.30, 5),      # scale down to ceil(10 * 0.3 / 0.6) = 5
])
def test_hpa_desired_replicas_match_kubernetes(r, rho, expected):
    hpa = HPA(0.6, 1)
    hpa.reset(1, CFG)
    assert int(hpa.desired(_ctx(r, rho))[0]) == expected


def test_hpa_uses_the_replica_count_the_metric_was_measured_on():
    """Utilisation 0.9 was measured on 10 replicas; 6 more have just become ready. Kubernetes sizes
    the demand from the 10 (ceil(10 * 0.9 / 0.6) = 15 <= 16 ready: no change), not from the 16."""
    hpa = HPA(0.6, 1)
    hpa.reset(1, CFG)
    ctx = _ctx(16, 0.9)
    ctx.golden["replicas"] = np.array([10])
    assert int(hpa.desired(ctx)[0]) == 15
    assert int(hpa.propose(ctx)[0, 0]) == NOOP
    del ctx.golden["replicas"]                               # the v0.1 behaviour: 16 x 0.9 / 0.6 -> 24, scale up again
    hpa.reset(1, CFG)
    assert int(hpa.desired(ctx)[0]) == 24 and int(hpa.propose(ctx)[0, 0]) == UP6


def test_hpa_scale_down_stabilisation_window():
    hpa = HPA(0.6, 3)
    hpa.reset(1, CFG)
    hpa.desired(_ctx(10, 0.9))                      # recommends 15
    assert int(hpa.desired(_ctx(10, 0.3))[0]) == 10  # would be 5, held at current by the recent 15



def test_harness_records_every_candidate_and_the_request():
    """On an incident day the runbook proposes failover first; the record must keep the needs-approval
    verdict of that candidate and mark the request, not just the action that ran."""
    from noisefloor.agents.rules import Runbook
    from noisefloor.eval.harness import run_episodes
    from noisefloor.sim.loads import make_profile
    profiles = [make_profile("incident", np.random.default_rng([5, i]), CFG) for i in range(2)]
    out = run_episodes(Runbook(0.7, 1), profiles, [1, 2], np.full(2, 10), CFG, Gate.load("gate_v1.json"), keep_decisions=True)
    requests = [d for d in out.decisions if d["requested"]]
    assert requests, "the runbook never filed a failover request on an incident day"
    d = requests[0]
    assert d["candidates"][0] == {"action": "failover", "verdict": "needs-approval", "reason": d["candidates"][0]["reason"]}
    assert d["proposed"][0] == "failover" and d["executed"] != "failover"
    granted = [d for d in out.decisions if d["approval"] == "granted"]
    assert granted and granted[0]["executed"] == "failover"
    assert sum(r.requests for r in out.results) >= len(granted) and sum(r.failovers for r in out.results) == len(granted)


def test_replay_applies_the_new_catalogs_action_budget():
    from noisefloor.audit.replay import replay_ledger

    class Env:
        def __init__(self, seq, executed):
            self.seq, self.data = seq, {"executed": executed, "args": {"replicas_after": 12}, "approval": None}

    class Ledger:
        def runs(self):
            return ["r"]

        def envelopes(self, rid):
            return [Env(s + 1, "scale_up_2") for s in range(8)]

    rep = replay_ledger(Ledger(), Gate.load("gate_v1.json"))       # budget 6 per 12 steps
    assert rep["executed_non_noop"] == 8 and rep["flips"] == 2
    assert all("budget" in f["reason"] for f in rep["examples"])


# ------------------------------------------------------------------------------------------ stats
def test_bootstrap_interval_covers_the_true_mean():
    rng = np.random.default_rng(0)
    hits = 0
    for rep in range(200):
        x = rng.normal(1.0, 1.0, (5, 16)) + rng.normal(0, 0.3, (5, 1))
        b = hierarchical_bootstrap(x, n_boot=800, seed=rep)
        hits += b["lo"] <= 1.0 <= b["hi"]
    assert hits / 200 > 0.85


def test_tost_holm_iqm_and_seed_interval():
    rng = np.random.default_rng(1)
    assert tost(rng.normal(0.0, 0.01, (5, 16)), margin=0.03)["equivalent"]
    assert not tost(rng.normal(0.08, 0.01, (5, 16)), margin=0.03)["equivalent"]
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert iqm([1, 2, 3, 4, 100, 5, 6, 7]) == pytest.approx(4.5)
    t = seed_t_interval(np.array([[1.0, 1.2], [0.9, 1.1], [1.1, 1.0], [1.0, 0.9], [1.2, 1.0]]))
    assert t["lo"] < 1.04 < t["hi"]


# ------------------------------------------------------------------------------------------ audit
def test_envelope_shape_and_ids():
    rid = run_id("jepa", "main", 3, "main", 0.8, 100004)
    assert rid == "jepa-main-s3-main-d80-z16-e100004"
    assert run_id("recon", "ood", 0, "isotropic", 0.8, 200001, z_dim=32) == "recon-ood-s0-isotropic-d80-z32-e200001"
    d = envelope_dict(rid, 0, 100004, {"proposed": ["scale_up_2"], "executed": "scale_up_2", "verdict": "allow",
                                       "effect": "reversible", "args": {"replicas_after": 12}}, sha="abc1234")
    assert d["seq"] == 1 and d["type"] == "decision" and d["policy_id"] == "gate-v1"
    assert d["producer"] == "noisefloor/agent@abc1234" and d["ts"].endswith("Z")


@pytest.mark.skipif(not HAVE_STERNWATCH, reason="sternwatch (audit extra) not installed")
def test_watchledger_ingest_is_idempotent():
    from sternwatch.bus import MemoryBus
    from sternwatch.ledger import WatchLedger
    rec = EnvelopeRecorder("jepa", "main", 0, "main", 0.8, [100000, 100001], sha="abc1234")
    for t in range(5):
        for i in range(2):
            rec(t, i, {"proposed": ["noop"], "executed": "noop", "verdict": "allow", "effect": "reversible"})
    bus = MemoryBus()
    rec.publish(bus)
    rec.publish(bus)                                 # the same envelopes twice
    ledger = WatchLedger()
    stats = ledger.ingest(bus, "noisefloor.decisions")
    assert stats.inserted == 10 and stats.duplicates == 10 and stats.invalid == 0
    assert ledger.count() == 10 and ledger.gaps(rec.ids[0]) == []

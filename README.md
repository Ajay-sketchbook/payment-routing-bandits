# ⚡ Adaptive Payment Gateway Routing with Multi-Armed Bandits

A payment orchestrator must decide, for **every single transaction**, which
gateway (bank/PSP) to send it to. Gateway success rates are never known exactly,
they drift during the day, and gateways go down without warning. A fixed
"always use PG-A" rule loses thousands of payments during an outage.

This project treats routing as a **multi-armed bandit** problem: each gateway is
an arm, each transaction is a pull, success = reward 1. The router learns
online from success/failure feedback and moves traffic as conditions change.

Everything — simulator, algorithms, evaluation — is written **from scratch in NumPy**.

---

## Results

10 random seeds × 40,000 transactions, 5 gateways (88–94% normal success),
two outages: **PG-A** (the best gateway) drops to 60% at txn 15,000 for 5,000 txns,
**PG-B** drops to 55% at txn 32,000 for 4,000 txns.

| Policy | Success rate | During outages | Traffic to failed gateway | Recovery (txns, median) | Cum. regret | Extra successes / 1M vs static |
|---|---|---|---|---|---|---|
| UCB1 | 93.18% ± 0.38% | 91.37% | 4.1% | 276 | 432 | +33,202 |
| Epsilon-Greedy (W=1000) | 93.11% ± 0.37% | 91.55% | 2.5% | 365 | 450 | +32,552 |
| **Thompson Sampling (W=1000)** | 92.80% ± 0.33% | **91.61%** | **1.6%** | **209** | 570 | +29,397 |
| UCB1 (W=1000) | 91.62% ± 0.22% | 90.42% | 2.2% | 200 | 1,058 | +17,610 |
| Epsilon-Greedy | 91.51% ± 0.65% | 86.27% | 19.1% | 1575 | 1,096 | +16,555 |
| Thompson Sampling | 91.36% ± 0.54% | 85.84% | 18.8% | 2326 | 1,156 | +15,092 |
| Static priority | 89.86% ± 0.31% | 74.99% | 55.6% | never | 1,751 | +0 |
| Random split | 89.26% ± 0.23% | 83.62% | 20.2% | never | 1,993 | −5,960 |

**Key findings**
- Every bandit beats static priority routing; the best add **~3.3 percentage points**
  of success overall and **~16.6 points during outages** (91.6% vs 75.0%).
- **Sliding windows matter for Thompson Sampling and ε-Greedy:** the plain versions keep
  trusting a failed gateway because of thousands of old successes (≈19% of outage traffic
  still goes to it, recovery ≈1,500–2,300 txns). With a 1,000-txn window that falls to
  1.6–2.5% and recovery to ≈200–370 txns.
- **Sliding-window UCB1 is worse than plain UCB1** here: with only 1,000 observations the
  exploration bonus stays large, so it keeps spending traffic on weaker gateways.
  Window size and exploration strength have to be tuned together.
- The API uses **sliding-window Thompson Sampling**: slightly below the top on average, but
  the best during outages and the least traffic sent to a dead gateway, which is
  what matters most for payments.

| | |
|---|---|
| ![rolling](reports/figures/rolling_success_rate.png) | ![split](reports/figures/traffic_split.png) |
| ![bars](reports/figures/policy_comparison.png) | ![regret](reports/figures/cumulative_regret.png) |

---

## Project structure

```
payment-routing-bandits/
├── router/
│   ├── gateways.py     ← simulator: drifting success rates + scheduled outages
│   ├── policies.py     ← Static, Random, ε-Greedy, UCB1, Thompson (+ sliding window)
│   ├── simulate.py     ← run a policy, compute regret / recovery / outage metrics
│   └── service.py      ← thread-safe routing service (route + feedback)
├── api/app.py          ← FastAPI: /route, /feedback, /stats, /simulate
├── benchmark.py        ← multi-seed comparison → reports/ (tables + plots)
├── tests/test_router.py
└── reports/            ← results.md, results.json, figures/
```

## How to run

Needs Python 3.10+.

```bash
cd payment-routing-bandits
python -m venv venv
venv\Scripts\activate              # Windows
# source venv/bin/activate         # macOS / Linux
pip install -r requirements.txt

python benchmark.py                # full benchmark, ~3 min  (or: python benchmark.py --quick)
python -m pytest -q                # tests
uvicorn api.app:app --reload       # API → open http://127.0.0.1:8000/docs
```

**Continuous integration:** `.github/workflows/tests.yml` runs the unit tests and a
quick 3-seed benchmark on Python 3.10 and 3.12 for every push and pull request
(GitHub → *Actions* tab shows each run).

**API demo:** in `/docs`, call `POST /simulate` with `n=5000` a few times and check
`GET /stats` after each call. Around 15,000 total transactions PG-A goes down and
its `attempts_in_window` falls towards zero as the router moves traffic away.

| Endpoint | Purpose |
|---|---|
| `POST /route` | returns `routing_id` + chosen gateway (+ decision latency) |
| `POST /feedback` | `{routing_id, success}` – the router learns from the outcome |
| `GET /stats` | per-gateway attempts and observed success rate in the window |
| `POST /simulate?n=` | pushes `n` simulated payments through the router |

---

## How the algorithms work

- **ε-Greedy** – send to the best observed gateway; with probability ε pick a random one.
- **UCB1** – score = observed rate + c·√(2 ln N / nᵢ); rarely-tried gateways get a bonus.
- **Thompson Sampling** – Beta(successes+1, failures+1) belief per gateway; sample each
  belief and route to the highest sample. Exploration fades as beliefs sharpen.
- **Sliding window** – only the last W outcomes count, so old evidence about a gateway
  that has since failed (or recovered) is forgotten.

**Evaluation:** every policy faces the *same* drift path and outage schedule for a given
seed (common random numbers), so differences come only from routing decisions.
Regret = (best true success probability − chosen gateway's true probability), summed.

## Limitations & next steps
- Feedback arrives instantly here; real gateways reply with delay (batched / delayed updates).
- One global policy; real routing is contextual (payment method, bank, amount) →
  **contextual bandits** (LinUCB) per segment.
- Window size is fixed; an adaptive change-point detector could reset only when a shift is detected.
- `pending` routing ids should expire after a timeout in production.

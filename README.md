# Biologically Plausible Secure MPC for Efficient Privacy-Preserving AI Training II

**Secure training of a Spiking Neural Network with a local learning rule (STDP) under Multi-Party Computation.**

![Python](https://img.shields.io/badge/python-3.12-3776ab?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-ee4c2c?logo=pytorch&logoColor=white)
![MP-SPDZ](https://img.shields.io/badge/MPC-MP--SPDZ-5b5ea6)
![Stage](https://img.shields.io/badge/stage%201%20(plaintext)-complete-2ea44f)
![MPC](https://img.shields.io/badge/MPC-secure%20STDP%20image-lightgrey)

B.Sc. thesis project, **Topic II**. This repository holds the full experimental code, from a plaintext
reference implementation (done) towards a secure MPC prototype (next).

> **Status.** Stage 1 (plaintext validation) is complete and reproducible. The MPC work has started: the three building
> blocks (add, multiply, compare) are explained and measured (dashboard Phase 5); the Phase 2 LIF layer's forward pass runs
> on secret shares in MP-SPDZ (Phase 6); and one whole STDP training image, including a secret winner-take-all, runs on
> shares and follows the plaintext trainer (Phase 7, see [Secure STDP image](#secure-stdp-image-phase-7)). **Training a
> network under MPC is not done**: only single images were checked, from a given state, and the cost per image is far too
> high for a full epoch yet. Nothing here makes a security claim about the final system. The remaining stages are in the
> [roadmap](#roadmap).

---

## Research question

Spiking neural networks are *sparse*: few neurons spike per timestep. It is tempting to conclude that
sparse computation is automatically cheaper under MPC. **That is an assumption, not a result.**
Under MPC, data-dependent control flow (which neuron spiked, when, how many) can leak information, and
hiding it (fixed-length, oblivious execution) can remove the very savings sparsity promises.

This project asks, with measurements:

> Does an SNN trained with a local, biologically plausible rule (STDP) actually reduce MPC cost compared
> with a conventional ANN **after** accounting for leakage protection, communication and secure control flow?

A negative answer is a valid outcome of this thesis.

## What exists today (Stage 1, plaintext)

A complete, tested, from-scratch pipeline on MNIST. Every operation of the LIF layer and the learning rule
is written explicitly (no framework magic) because each one has to be re-implemented as a secure operation later.

```mermaid
flowchart LR
    A[MNIST image<br/>784 pixels] -->|rate encoding<br/>fixed T = 25| B[Spike train<br/>25 x 784]
    B --> C[LIF hidden layer<br/>100 neurons<br/>winner-take-all]
    C -->|trace-based STDP<br/>unsupervised, local| C
    C -->|spike counts / T| D[Linear readout<br/>softmax, supervised]
    D --> E[Digit 0-9]
```

| Component | File | What it does |
|---|---|---|
| Config | [configs/experiment_config.yaml](configs/experiment_config.yaml) | Single source of truth for every hyperparameter |
| Encoding | [src/encoding.py](src/encoding.py) | Rate coding: a pixel spikes with probability equal to its intensity, for a fixed `T` |
| LIF layer | [src/lif_model.py](src/lif_model.py) | `u = beta*u + I`, threshold compare, reset, winner-take-all, adaptive threshold |
| STDP | [src/stdp.py](src/stdp.py) | Trace-based local rule; only the winning neuron updates, using only its own spike and its inputs' traces |
| Readout | [src/readout.py](src/readout.py) | Linear softmax classifier, the only step that uses labels; also precision / recall / F1 |
| ANN baseline | [src/ann_model.py](src/ann_model.py) | 784 -> 100 (ReLU) -> 10, Adam + backprop |
| Dashboard | [src/app_server.py](src/app_server.py), [src/web/index.html](src/web/index.html) | Local web UI that runs the real code above, phase by phase |

### The model in three equations

```
Membrane   u_i[t] = beta * u_i[t-1] + sum_j W[j,i] * s_j[t]
Spike      s_i[t] = 1  if  u_i[t] >= threshold + theta_i,  else 0       (then reset)
STDP       W[:,j] += lr * (trace - x_target) * (w_max - W[:,j])^mu      (winner j only)
```

`theta_i` is a per-neuron adaptive threshold that rises each time neuron `i` wins, which keeps the neurons
from all learning the same digit. After each image, every neuron's incoming weights are rescaled to a fixed sum.

## Results (plaintext, MNIST)

Measured on the full official 10,000-image test set. Training uses 5,000 images (validation: 1,000),
fixed seeded split, **`seed: 0`**, CPU (4 threads), default values from the config file.

| Model | Learning | Test accuracy | Training time |
|---|---|---:|---:|
| ANN 784-100-10 (ReLU) | backprop, supervised | **92.96 %** | 1.6 s (20 epochs) |
| SNN: STDP hidden layer + linear readout | local STDP (unsupervised) + supervised readout | **80.90 %** (macro-F1 80.6 %) | 13.1 s STDP (1 epoch) + 6.5 s readout |
| Control: *untrained* random hidden layer + same readout | readout only | 55.68 % | n/a |

How to read this:

- **The control matters.** It isolates what STDP contributes: +25 points over a random hidden layer with
  an identical readout. Without it, the SNN number would not show that anything was learned.
- **The ANN wins on accuracy here**, and this is expected: it is trained end-to-end with labels. The thesis
  is not about beating the ANN in plaintext; it is about what each approach costs when run securely.
- Most SNN errors are between visually similar digits (4 vs 9, 3/5/8). The unsupervised layer groups
  look-alikes together and the readout cannot always separate them.
- **These numbers are preliminary.** The STDP hyperparameters (`learning.*`) have not been tuned; the SNN was
  trained for a single epoch. The ANN over-fits the small training set (99.5 % train vs 91.8 % validation).
- Not yet measured: variance over multiple seeds. Treat single-run differences of ~1 point as noise.

### SNN activity and plaintext operation counts (per image)

| Quantity | Value |
|---|---:|
| Input spikes (T = 25) | ~2,597 |
| Hidden spikes (winner-take-all) | ~4.0 |
| Hidden neurons that ever fired | 99 / 100 |
| Dense synaptic additions (every weight, every step) | 1,960,000 |
| Event-driven synaptic additions (only where an input spiked) | 259,740 |
| ANN multiplications (for comparison) | 79,400 |

> **Caution.** These are *plaintext* operation counts. They are **not** an MPC cost. Under MPC the
> event-driven count is not directly usable, because skipping work for silent inputs reveals which inputs
> were silent. Whether sparsity helps securely is exactly what Stages 3-6 will measure.

## Secure LIF forward pass (Phase 6)

The Phase 2 hidden layer (784 inputs, 100 LIF neurons, 25 timesteps, `beta = 0.9`, threshold 50, subtract-reset) computed
by three MP-SPDZ parties on secret shares: party 0 holds the image spikes, party 1 the weights, party 2 only computes.
Replicated ring protocol (3 parties, honest majority, semi-honest), 64-bit ring, fixed point with 16 fractional bits.
All parties run on one machine, so **rounds and bytes** are meaningful, wall-clock time is not.

| Check | Result |
|---|---|
| Spikes that differ from the plaintext layer | 0 of 2,500 on the two images tried (#0: 167 spikes, #3: 546 spikes) |
| Largest membrane error | about 0.003 (the fixed-point step is 0.000015; the error accumulates through the leak) |

| One timestep, all 100 neurons | Rounds | Data sent (all 3 parties) |
|---|---:|---:|
| Synaptic current `W·s` | 4 | 120 KB |
| Leak `beta·u` | 3 | 116 KB |
| Threshold `u >= theta` | 9 | 214 KB |
| Reset `u - theta·spike` | 0 | 0 (theta = 50 is a whole number) |
| **Whole forward pass, 25 timesteps** | **314** | **11.3 MB** (plus 0.8 MB once to share the inputs) |

What this shows, and what it does not:

- **The cost does not depend on the input.** The two images have 167 and 546 spikes and sent exactly the same number of
  bytes: the secure program does the same work whatever the spikes are. The sparsity that makes the plaintext layer cheap
  gives no saving here. This is the central question of the thesis, answered for one layer and one protocol.
- **The Phase 5 estimates were off.** Multiplying by the public `beta` was estimated as free; it costs as much as the
  matrix product because every fixed-point product must be truncated. A fractional public theta (1.5) also makes the
  reset cost communication (3 rounds, about 82 KB per timestep in a quick check). That is specific to how the Phase 6
  program multiplies (`sfix` times a constant); multiplying the raw integer instead, as Phase 7 does for `theta_plus`, is local.
- MP-SPDZ itself suggests `program.use_trunc_pr = True` to make truncation cheaper. That is the first optimisation candidate
  for Stage 5 and is **not** applied.
- The validation run opens every spike and membrane value to compare with plaintext. A real deployment would never open
  them; the cost numbers come from separate runs that do not.
- This is the forward pass only. Winner-take-all (an argmax) and the STDP update with a secret winner are still to be built, and
  the Phase 5 estimates say they will dominate.

## Secure STDP image (Phase 7)

One STDP training image (25 timesteps) computed by the three MP-SPDZ parties on secret shares, starting from a **copy** of the
trained layer: trace update, synaptic current, leak, threshold test with a secret adaptive theta, **winner-take-all with a secret
winner**, the weight update of the winner's column, theta update, reset, and the per-neuron rescaling at the end. It mirrors
`STDPTrainer.train_sample` and is checked against it. Same setting as Phase 6 (replicated ring, 64-bit, 16 fractional bits).

| Check (images 0, 1, 2 of the test set) | Result |
|---|---|
| Timesteps with a different winner than plaintext | 0 of 25, on each of the three images (3, 4 and 2 winning steps) |
| Largest weight error after the image | at most 0.00007 (the fixed-point step is 0.000015) |
| Threshold (theta) error | 0 |

| One image | Rounds | Data sent (all 3 parties) |
|---|---:|---:|
| **Whole image** | **5,655** | **312.5 MB** |
| of which rescaling to `w_norm` (once) | 2,362 | 96 MB |
| of which keeping weights in [0, w_max] (25 timesteps) | 28 per step | 85 MB |
| of which writing the winner's column back + its change | | 94 MB |
| of which input trace | | 23 MB |
| of which winner-take-all | 74 per step | 11 MB |

- **The cost is identical for every input** (5,655 rounds and 312,480,032 bytes on all three images, which have different
  numbers of winning steps; the unit tests also get identical costs for all-zero, all-one and random spikes). The sparsity that
  makes plaintext STDP cheap (about 4 neurons updated per image) gives no saving here, because the winner must stay secret
  and so every weight is touched. This is the thesis question, answered for one layer and one protocol.
- **Two savings, measured.** (1) Spikes are 0/1 integers, so the synaptic current as an integer times raw fixed-point weights
  needs no truncation: 2.4 KB per timestep instead of 120 KB with fixed-point matrices (Phase 6 used the expensive form).
  (2) Clamping only the winner's column of 784 values instead of all 78,400 weights: 3.4 MB instead of 339 MB per timestep.
  Both change the cost, not the result.
- **Truncation: speed or secrecy.** At compile time MP-SPDZ warns that its default probabilistic truncation "leaks some
  information" (eprint 2024/1127) and offers `sfix.round_nearest = True`. Every number above uses the default. The exact
  version costs 1.9x the data and 2.6x the rounds for the whole image (596 MB, 14,562 rounds). Whether that leakage matters for the
  threat model is not decided here; it belongs to the security analysis and needs the supervisor.
- **Not done yet.** Only single images from a given state were checked. At 312 MB per image a full epoch of 5,000
  images would send about 1.6 TB between the parties, so the cost has to come down (Stage 5) before secure training of
  the whole network is realistic. The supervised readout is not secure yet. The validation run opens the final weights and the
  winners; the cost numbers come from separate runs that do not.

## Quick start

Requires Python 3.12 and about 65 MB of disk for MNIST. CPU is sufficient.

```bash
git clone --recurse-submodules https://github.com/Pedro4O4/Biologically-Plausible-Secure-MPC-for-Efficient-Privacy-Preserving-AI-Training.git
cd Biologically-Plausible-Secure-MPC-for-Efficient-Privacy-Preserving-AI-Training

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_mnist.py        # one-time dataset download into data/
python -m unittest discover -s tests -v # 46 tests, about half a minute
python src/app_server.py                # dashboard on http://127.0.0.1:8000
```

`--recurse-submodules` is only needed for the MP-SPDZ checkout used in the later stages; the plaintext
pipeline does not depend on it.

### Using the dashboard

Open <http://127.0.0.1:8000>. The dashboard grows one phase at a time; each tab runs the real project code.

| Tab | What you can do |
|---|---|
| Phase 0 · Encoding | Inspect an MNIST digit and its spike raster; change the sample and timestep |
| Phase 1 · ANN | Train the backprop baseline and watch accuracy, per-digit recall |
| Phase 2 · LIF | Watch the hidden layer spike over time; click a neuron to inspect it; preview leak and threshold |
| Phase 3 · STDP | Train unsupervised; watch the 100 neurons' weights turn into digit-like features |
| Phase 4 · Readout | Train the readout; compare ANN vs SNN vs control; confusion matrix; operation counts |
| Phase 5 · Secret sharing | Step through add / multiply / compare under 3-party replicated secret sharing with real numbers (what each party holds, computes and sends); see what one party can see; run the same operations in real MP-SPDZ and compare rounds and bytes |
| Phase 6 · Secure LIF | Run the Phase 2 hidden layer on secret shares (3 local MP-SPDZ parties): plaintext vs secure spikes and membrane side by side, the six steps of one neuron's timestep, measured cost per operation, precision, and the gap to the Phase 5 estimates |
| Phase 7 · Secure learning | Run one STDP training image on secret shares (winner-take-all with a secret winner, the weight update, the rescaling) on a copy of your trained layer: plaintext vs secure winners and learned weights, cost of every operation, two measured savings, the truncation trade-off |

**Training never starts on its own**; it only runs when you press the Train button. Phase 5's "Run in real MP-SPDZ" button needs MP-SPDZ built (`replicated-ring-party.x`); the dashboard tells you if something is missing. It starts 3 local parties and no training. A different port can be
passed as the first argument: `python src/app_server.py 8001`. Trained models and metrics are saved in
`results/` and reloaded on the next start (that folder is git-ignored).

## Configuration

All experiment parameters live in [configs/experiment_config.yaml](configs/experiment_config.yaml) and are
read by every script. Nothing is hard-coded in the model files. Key settings:

| Key | Default | Meaning |
|---|---|---|
| `simulation.num_steps` | 25 | Fixed time window `T`. Independent of the input by design (see [Security notes](#security-and-leakage-notes)) |
| `architecture` | 784 / 100 / 10 | Inputs / hidden LIF neurons / classes |
| `seed` | 0 | Controls the data split and weight initialisation |
| `neuron.beta`, `neuron.threshold` | 0.9, 50.0 | Leak and spike threshold (Phase 2) |
| `learning.*` | see file | STDP rate, trace decay, soft bound, threshold bonus (Phase 3) |
| `readout.*` | see file | Readout optimiser settings (Phase 4) |

To reproduce the table above, use `seed: 0` and the committed defaults, then press Train in Phases 1, 3 and 4.

## Repository layout

```
configs/    experiment_config.yaml        all hyperparameters
src/        encoding, lif_model, stdp, readout, ann_model, data, config_loader
            mpc_sim.py, mpc_real.py       secret-sharing simulation + real MP-SPDZ runner (Phase 5)
            mpc_lif.py                    secure LIF layer: runner, check against plaintext, cost (Phase 6)
            mpc_stdp.py                   secure STDP image: runner, check against STDPTrainer, cost (Phase 7)
            app_server.py + web/          local dashboard
mpc/        ops_demo.mpc, secure_lif.mpc, secure_stdp.mpc, hello_mpc.mpc   MP-SPDZ programs (compiled from here, the submodule stays clean)
tests/      test_lif, test_stdp, test_readout, test_mpc_sim, test_mpc_lif, test_mpc_stdp   (unit tests, 46 total)
scripts/    download_mnist.py, snntorch_sanity_check.py
MP-SPDZ/    MPC framework (git submodule; used from Stage 2 onwards)
results/    generated models and metrics (git-ignored)
```

## Roadmap

The thesis follows six stages. Stage 1 is split into the dashboard's Phases 0-4.

| Stage | Goal | Status |
|---|---|---|
| 1 | Plaintext validation: dataset, encoding, LIF, STDP, readout, ANN baseline | **Done** (hyperparameters not yet tuned) |
| 2 | Secure baseline: small conventional NN under MPC | Not started |
| 3 | Microbenchmarks of secure SNN parts: membrane update, threshold, reset, local weight update | Done for one image: every part (current, leak, threshold, winner-take-all, weight update, rescaling) runs on shares and is measured (Phases 6-7) |
| 4 | End-to-end secure SNN training prototype | Not started |
| 5 | Profiling, bottleneck analysis, at least one measured optimisation | Not started |
| 6 | Comparison: plaintext ANN / plaintext SNN / secure ANN / secure SNN (accuracy, time, communication, rounds, leakage) | Not started |

Optional extensions (only if time allows): Fashion-MNIST, N-MNIST, surrogate-gradient SNN reference,
a second MPC protocol, hardware energy measurement. Energy is only reported if a real measurement setup exists.

## Security and leakage notes

These are design constraints for the secure stages, recorded here so the plaintext code stays compatible with them.

- **Fixed-length execution.** The simulation always runs exactly `T` steps, whatever the input. The amount of
  computation therefore does not reveal anything about the input.
- **Data-dependent sparsity can leak.** When spikes occur, where they occur, and how many occur are all
  potentially sensitive. A secure implementation must either hide them (oblivious handling, which costs
  communication and computation) or document them explicitly as released information.
- **Candidate setting, not a claim.** The initial MPC candidate is MP-SPDZ with a 3-party, honest-majority,
  semi-honest protocol. The final protocol, party count and adversary model will be fixed after feasibility
  testing and supervisor approval. No stronger security than the chosen protocol provides will be claimed.
- **Public vs private** values (inputs, membrane potentials, spike events, weight updates, released outputs)
  will be listed explicitly once the secure prototype exists.

## Tech stack

Python 3.12, PyTorch (CPU), PyYAML, matplotlib, MP-SPDZ. The LIF layer is written by hand; snnTorch is used
only as an optional cross-check in `scripts/snntorch_sanity_check.py`. The dashboard uses only the Python
standard library and a single static HTML page (no framework, no build step).

## Author

Ebram Nageh ([@Pedro4O4](https://github.com/Pedro4O4)). B.Sc. Computer Science (IT Security), German International University (GIU).

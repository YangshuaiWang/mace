# Repo Agent Instructions (Codex)

You are an AI coding agent working in this repository. Follow these instructions with highest priority.

## Project Goal
Implement and iterate on **IB-UQ-MACE**: add an IB-UQ confidence-aware encoder and OOD gating to MACE, while strictly preserving E(3) equivariance.

## Hard Constraints (MUST)
1) **Do NOT modify the MACE backbone / message passing.**
   - No changes to neighborhood construction, interaction blocks, tensor products, or equivariant convolutions.
   - Only modify/add code in the **head/readout** and training plumbing if needed.

2) **Equivariance constraint for latent injection**
   - Any stochastic latent `z` and gate `m(x)` may be computed from **L=0 (scalar invariant) features only**.
   - `z` may be injected **ONLY** into **L=0 invariant channels** before the equivariant readout.
   - NEVER inject `z` into L>0 equivariant features directly.
   - Forces must remain E(3)-equivariant: F(RX) = R F(X).

3) **IB-UQ mechanism must be implemented exactly**
   For each atom/environment:
   z = diag(m(x)) * z_bar(x) + diag(1 - m(x)) * z0,  z0 ~ N(0, I)
   m(x) = sigmoid(g(x)) in [0,1]^{d_z}
   z_bar(x) learned code

4) **Outputs**
   - Preserve original outputs (energy, forces) API.
   - Additionally expose a gate-based OOD score:
     gate_score = 1 - mean(m) (per-structure preferred; per-atom acceptable too).
   - Provide deterministic mode (fixed z0 or z0=0) for tests and reproducibility.

## Preferred Implementation Strategy
- Add a `ConfidenceGate` module:
  - Input: per-atom L=0 scalars h0 [n_atoms, d0]
  - Output: m [n_atoms, d_z], z_bar [n_atoms, d_z]
- Inject latent into L=0 scalars via either:
  - Concatenation: h0_aug = MLP([h0 || z]) -> new scalars, or
  - FiLM: h0_aug = a(z) ⊙ h0 + b(z)
- Keep changes minimal and aligned with existing code style.

## Testing & Verification (MUST)
After changes, run at least one of the following and ensure it passes:
- `pytest` (if available), or
- `python scripts/quick_check_ib_uq.py`

Minimum checks:
- forward pass returns energy and forces with correct shapes
- gate_score exists and lies in [0,1]
- deterministic mode produces identical outputs across runs

## Workflow Rules
- First: search and identify the relevant head/readout code paths (file list + symbols).
- Second: propose a concrete plan (files to touch, classes/functions to add).
- Third: implement in small commits/steps.
- Fourth: run tests / quick check and report results.

## Documentation
- Add docstrings explaining:
  - the IB-UQ gating equation and interpretation
  - why injecting only into L=0 preserves equivariance

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import iread


def _parse_key_list(raw: str) -> list[str]:
    return [key.strip() for key in raw.split(",") if key.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect extxyz labels and run sanity checks on energy/forces/stress targets.",
    )
    parser.add_argument("--extxyz", type=Path, required=True, help="Path to input extxyz file")
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--energy_keys", default="energy", help="Comma-separated candidate keys")
    parser.add_argument("--forces_keys", default="forces", help="Comma-separated candidate keys")
    parser.add_argument("--stress_keys", default="stress", help="Comma-separated candidate keys")
    parser.add_argument("--require_forces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_energy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_stress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def _collect_frames(extxyz: Path, max_frames: int):
    frames = []
    for idx, atoms in enumerate(iread(extxyz, index=":")):
        if idx >= max_frames:
            break
        frames.append(atoms)
    return frames


def _choose_matching_key(candidates: list[str], available: set[str]) -> str | None:
    for key in candidates:
        if key in available:
            return key
    return None


def _extract_energy(atoms, key: str | None):
    if key is None:
        return None
    if key in atoms.info:
        return float(np.asarray(atoms.info[key], dtype=float).reshape(-1)[0])
    if atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
        return float(np.asarray(atoms.calc.results[key], dtype=float).reshape(-1)[0])
    return None


def _extract_forces(atoms, key: str | None):
    if key is None:
        return None
    if key in atoms.arrays:
        return np.asarray(atoms.arrays[key], dtype=float)
    if key in atoms.info:
        return np.asarray(atoms.info[key], dtype=float)
    if atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
        return np.asarray(atoms.calc.results[key], dtype=float)
    return None


def _extract_stress(atoms, key: str | None):
    if key is None:
        return None
    if key in atoms.info:
        return np.asarray(atoms.info[key], dtype=float)
    if key in atoms.arrays:
        return np.asarray(atoms.arrays[key], dtype=float)
    if atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
        return np.asarray(atoms.calc.results[key], dtype=float)
    return None


def validate_extxyz_labels(args: argparse.Namespace) -> tuple[int, dict]:
    if not args.extxyz.exists():
        raise FileNotFoundError(f"Input extxyz does not exist: {args.extxyz}")

    frames = _collect_frames(args.extxyz, args.max_frames)
    if not frames:
        raise ValueError(f"No frames found in extxyz: {args.extxyz}")

    natoms = np.array([len(at) for at in frames], dtype=int)
    elements = sorted({symbol for atoms in frames for symbol in atoms.get_chemical_symbols()})

    all_array_keys = sorted(set().union(*(atoms.arrays.keys() for atoms in frames)))
    all_info_keys = sorted(set().union(*(atoms.info.keys() for atoms in frames)))
    all_calc_keys = sorted(set().union(*(getattr(getattr(atoms, "calc", None), "results", {}).keys() for atoms in frames)))

    energy_candidates = _parse_key_list(args.energy_keys)
    forces_candidates = _parse_key_list(args.forces_keys)
    stress_candidates = _parse_key_list(args.stress_keys)

    energy_key = _choose_matching_key(energy_candidates, set(all_info_keys) | set(all_calc_keys))
    forces_key = _choose_matching_key(forces_candidates, set(all_array_keys) | set(all_info_keys) | set(all_calc_keys))
    stress_key = _choose_matching_key(stress_candidates, set(all_info_keys) | set(all_array_keys) | set(all_calc_keys))

    energy_per_atom = []
    force_magnitudes = []
    stress_magnitudes = []

    missing_counts = {"energy": 0, "forces": 0, "stress": 0}
    nonfinite_counts = {"energy": 0, "forces": 0, "stress": 0}

    for atoms in frames:
        energy = _extract_energy(atoms, energy_key)
        if energy is None:
            missing_counts["energy"] += 1
        else:
            if np.isfinite(energy):
                energy_per_atom.append(energy / max(len(atoms), 1))
            else:
                nonfinite_counts["energy"] += 1

        forces = _extract_forces(atoms, forces_key)
        if forces is None:
            missing_counts["forces"] += 1
        else:
            if np.isfinite(forces).all():
                force_magnitudes.extend(np.linalg.norm(forces.reshape(-1, 3), axis=1).tolist())
            else:
                nonfinite_counts["forces"] += 1

        stress = _extract_stress(atoms, stress_key)
        if stress is None:
            missing_counts["stress"] += 1
        else:
            flat = np.asarray(stress, dtype=float).reshape(-1)
            if np.isfinite(flat).all():
                stress_magnitudes.append(float(np.linalg.norm(flat)))
            else:
                nonfinite_counts["stress"] += 1

    stats = {
        "natoms_min": int(natoms.min()),
        "natoms_median": float(np.median(natoms)),
        "natoms_max": int(natoms.max()),
        "energy_per_atom_median": float(np.median(energy_per_atom)) if energy_per_atom else None,
        "force_magnitude_median": float(np.median(force_magnitudes)) if force_magnitudes else None,
        "force_magnitude_max": float(np.max(force_magnitudes)) if force_magnitudes else None,
        "stress_magnitude_median": float(np.median(stress_magnitudes)) if stress_magnitudes else None,
        "stress_magnitude_max": float(np.max(stress_magnitudes)) if stress_magnitudes else None,
    }

    required = {
        "energy": bool(args.require_energy),
        "forces": bool(args.require_forces),
        "stress": bool(args.require_stress),
    }
    matched_keys = {"energy": energy_key, "forces": forces_key, "stress": stress_key}

    errors = []
    expected = {
        "energy": energy_candidates,
        "forces": forces_candidates,
        "stress": stress_candidates,
    }

    for target in ["energy", "forces", "stress"]:
        if required[target] and matched_keys[target] is None:
            errors.append(
                f"Missing required {target} key. Expected one of {expected[target]}, "
                f"found info keys {all_info_keys}, array keys {all_array_keys}, and calc keys {all_calc_keys}."
            )
        if required[target] and missing_counts[target] > 0:
            errors.append(
                f"Required {target} key '{matched_keys[target]}' is missing in {missing_counts[target]}/"
                f"{len(frames)} sampled frames."
            )
        if required[target] and nonfinite_counts[target] > 0:
            errors.append(
                f"Required {target} key '{matched_keys[target]}' has NaN/Inf in {nonfinite_counts[target]} sampled frames."
            )

    warnings = [
        "Units are not inferred; reported values are raw magnitudes from the file.",
    ]
    if stats["force_magnitude_max"] is not None and stats["force_magnitude_max"] > 1.0e4:
        warnings.append("Force magnitudes are very large (>1e4); verify units and label scaling.")
    if stats["stress_magnitude_max"] is not None and stats["stress_magnitude_max"] > 1.0e5:
        warnings.append("Stress magnitudes are very large (>1e5); verify units and conventions.")

    report = {
        "extxyz": str(args.extxyz),
        "frames_read": len(frames),
        "max_frames": int(args.max_frames),
        "elements": elements,
        "available_keys": {"arrays": all_array_keys, "info": all_info_keys, "calc_results": all_calc_keys},
        "expected_keys": expected,
        "matched_keys": matched_keys,
        "required": required,
        "missing_counts": missing_counts,
        "nonfinite_counts": nonfinite_counts,
        "stats": stats,
        "warnings": warnings,
        "errors": errors,
        "ok": len(errors) == 0,
    }

    return (0 if report["ok"] else 2), report


def _print_summary(report: dict) -> None:
    print("=== extxyz label validation ===")
    print(f"File: {report['extxyz']}")
    print(f"Frames read: {report['frames_read']} (max_frames={report['max_frames']})")
    print(
        "Atoms/frame (min/median/max): "
        f"{report['stats']['natoms_min']}/{report['stats']['natoms_median']}/{report['stats']['natoms_max']}"
    )
    print(f"Elements: {', '.join(report['elements'])}")
    print(f"Info keys: {report['available_keys']['info']}")
    print(f"Array keys: {report['available_keys']['arrays']}")
    print(f"Calc result keys: {report['available_keys']['calc_results']}")

    print("Matched keys:")
    print(f"  energy: {report['matched_keys']['energy']}")
    print(f"  forces: {report['matched_keys']['forces']}")
    print(f"  stress: {report['matched_keys']['stress']}")

    stats = report["stats"]
    print("Magnitude stats (raw units):")
    print(f"  energy/atom median: {stats['energy_per_atom_median']}")
    print(
        f"  force |F| median/max: {stats['force_magnitude_median']} / {stats['force_magnitude_max']}"
    )
    print(
        "  stress |S| median/max: "
        f"{stats['stress_magnitude_median']} / {stats['stress_magnitude_max']}"
    )

    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")

    if report["errors"]:
        print("Errors:")
        for error in report["errors"]:
            print(f"  - {error}")
    else:
        print("Validation status: OK")


def main() -> int:
    args = parse_args()
    code, report = validate_extxyz_labels(args)
    _print_summary(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"Wrote JSON report to {args.out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

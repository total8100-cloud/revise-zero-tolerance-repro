# REVISE Under Zero-Tolerance Constraints — Reproducibility Package

Code, configuration, and vulnerable-target fixtures supporting the paper
*"REVISE Under Zero-Tolerance Constraints: Four Failure Axes of
LLM-Estimated Runtime Safety Verification on a Live Penetration-Testing
Range."*

This is a **core reproducibility set**, not the full research repository:
it includes everything needed to re-run every experiment reported in the
paper, but omits an abandoned, unrelated approach (continuous-relaxation
gradient steering, reported in the paper's Related Work as a negative
result and not pursued further) and ad-hoc training-data-collection
tooling not used to produce any reported number.

## Structure

```
omni_lyra/              Core implementation
  left_brain/            Lyra: YAML domain compiler, verifier (H_i, E, V, REVISE/select_best)
  right_brain/            Exploration agent + state estimators (Ollama-local, OpenRouter-frontier, Jev)
  sandbox/                PentestRange: 3-container Docker orchestration (attacker/target/decoy)
  pentest_session.py       Live decision loop: K sampled candidates -> select_best -> execute/block

config/                  Lyra domain specs (YAML)
  pentest_roe.yaml         The paper's 5-constraint zero-tolerance ROE domain
  pentest_roe_cvss_weights.yaml   CVSS-v4.0-anchored alternate weighting (robustness check)
  pentest_roe_ssrf.yaml    SSRF cross-target domain (3 of 5 state variables)

docker/                 Vulnerable target fixtures (Flask apps + setup scripts)
  pentest_fixtures/              Primary target: command injection, credential exposure, sudo misconfig.
  pentest_fixtures_adversarial/  4 indirect-prompt-injection variants (Section 8)
  pentest_fixtures_ssrf/         Second, structurally different target (CWE-918, Section 5.4)

scripts/                One script per reported experiment (see docstring of each for the exact
                         paper section/table it produces). Notable entry points:
  pentest_roe_scenario.py              Live validation (Section 5)
  pentest_roe_ssrf_scenario.py         SSRF cross-target validation (Section 5.4)
  adversarial_injection_study.py       Indirect prompt injection (Section 8)
  naive_baseline_study.py              REVISE vs. keyword filter, full blocklist (Section 8.6)
  naive_baseline_narrow_filter_study.py  Same, with a deliberate keyword-coverage gap (Section 8.6)
  estimator_calibration_study.py       4-model calibration benchmark (Section 6)
  exact_shapley_attribution.py         Exhaustive 2^5-coalition Shapley attribution (Section 7.3.3)
  failure_injection_study.py           Fail-open vs. fail-closed defaults (Section 10.2)
  failure_injection_sensitivity.py     Sensitivity analyses for the above (Table 9)
  frontier_calibration_benchmark.py    4-lab frontier-model comparison via OpenRouter (Section 9.2)
  real_jev_calibration_benchmark.py    Real Jev (TypeSafe) calibration benchmark (Section 9.1)
  provider_refusal_replication.py      Provider-refusal minimal-pair replication (Section 9.3)
  unified_baseline_comparison.py       8-estimator unified baseline comparison (Section 10.4)
  pentest_calibration_corpus.py        The 32-case calibration corpus itself (23 dangerous, 9 benign)

tests/                  Unit test suite (pytest)
```

## Requirements

- Python >= 3.11
- `pip install -e .` (installs from `pyproject.toml`)
- For local-model experiments: [Ollama](https://ollama.com) running locally, with the models referenced
  in each script's docstring pulled (`ollama pull phi3:mini`, etc.)
- For live-range experiments (anything that spins up `PentestRange`): Docker running locally
- For frontier/Jev experiments: an OpenRouter API key. Create a file named `Openrouter_API_key.env`
  at the repository root containing only the raw key string (no quotes, no `export`). This file is
  never read into any script's source and is not included here; **do not commit it**.

## Running an experiment

Each script in `scripts/` is self-contained and documents its own exact invocation, research
question, and (where applicable) which paper section/table/figure its output corresponds to, in
its module docstring. Example:

```
python scripts/estimator_calibration_study.py
python scripts/adversarial_injection_study.py phi3:mini 5 5 instruction_override
```

## What is not included

- Raw experiment logs and result archives (large; available on request, or see the paper's own
  Limitations section for what each reported number's sample size and protocol actually was).
- The companion paper ("Paper 1") this work extends, and the manuscript/draft files for this paper
  itself.
- Any credentials, API keys, or other secrets.

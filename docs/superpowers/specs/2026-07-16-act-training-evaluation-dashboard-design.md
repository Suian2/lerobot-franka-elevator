# ACT Offline Training Evaluation Design

Date: 2026-07-16

## Scope Decision

Version one is a small offline extension of
`hardware_test/evaluation/eval_act_offline.py`. It reads saved checkpoint
metadata, recomputes the ACT CVAE objective on fixed expert episodes, and adds
loss/checkpoint visualizations to the expert-comparison metrics that already
exist.

Version one does not modify the training loop and does not add local JSONL
training metrics. That future half of the hybrid design is deferred.

The tool supports only the current single-floor button-press models. The
conditioned multi-floor model and `observation.environment_state` are out of
scope.

## Goals

The offline report must show:

- checkpoint configuration: `use_vae`, `kl_weight`, optimizer and backbone
  learning rates, and `normalization_mapping`;
- saved normalization statistics for images, states, and actions;
- whether each requested metric is exactly stored, recomputed, inferred, or
  unavailable;
- recomputed normalized L1, unweighted KLD, weighted KLD, and total CVAE loss
  on the same fixed expert episodes;
- the existing expert-versus-model action, physical error, commanded-delta
  trajectory, gripper, and horizon metrics;
- visual comparison across selected checkpoints.

## Non-goals

- No changes under `src/lerobot/scripts/` or to the training process.
- No JSONL writer, distributed metric aggregation, resume handling, W&B change,
  or future-training dashboard.
- No attempt to reconstruct historical minibatches or original training loss.
- No multi-floor evaluation or environment-state handling.
- No robot execution, live server, database, or new dependency.
- No new composite model score or automatic claim of task success.
- No per-latent posterior-collapse analysis in version one.

## Recoverability Rules

Every displayed value carries one of four labels:

- `exact`: directly read from a saved artifact;
- `recomputed`: evaluated now using the declared dataset and protocol;
- `inferred`: implied by saved configuration under a displayed assumption;
- `unavailable`: not present in the available artifacts.

| Item | Status for current runs | Source or limitation |
| --- | --- | --- |
| `use_vae` | Exact | Checkpoint `train_config.json` |
| `kl_weight` | Exact | Checkpoint policy configuration |
| Optimizer/backbone LR | Exact at saved checkpoints | Configuration and optimizer param groups |
| Full LR history | Inferred constant for these runs | Scheduler is absent and saved groups retain `1e-5`; this is not an observed per-step history |
| `normalization_mapping` | Exact | Checkpoint configuration |
| Normalization statistics | Exact | Saved preprocessor statistics |
| Original total training loss | Unavailable | Not persisted by the current runs |
| Original training L1/KLD | Unavailable | Not persisted by the current runs |
| Original gradient norm | Unavailable | Not persisted by the current runs |
| Deployment normalized L1 | Recomputed | Evaluation mode and zero latent on fixed expert episodes |
| CVAE L1/KLD/weighted KLD/total | Recomputed | Action-conditioned CVAE replay on fixed expert episodes |
| Expert-versus-model physical metrics | Recomputed | Existing offline evaluator |

The report must never place recomputed CVAE values on a chart labelled
`historical training loss`.

The current 29-episode and 50-episode model directories do not prove their
original episode membership. Unless a verified holdout set is supplied, the
report labels the result `expert-data replay / original train membership
unknown`, not `validation` or `generalization`.

## Loss Replay

The calculations must match ACT's current loss definition. For normalized
expert actions `a`, predicted actions `a_hat`, valid non-padding elements,
posterior mean `mu`, and posterior log variance `log_var`:

```text
L1 = sum(abs(a_hat - a) over valid action elements)
     / number_of_valid_action_elements

KLD = mean_over_batch(
    sum_over_latent(-0.5 * (1 + log_var - mu^2 - exp(log_var)))
)

weighted_KLD = kl_weight * KLD
total = L1 + weighted_KLD
```

Serialized metric names use `kld_loss`, matching `ACTPolicy.forward`.

Two different evaluations remain separate:

1. **Deployment inference:** call `predict_action_chunk` in evaluation mode.
   ACT uses the zero latent, so this gives deterministic deployment L1 and the
   existing physical-unit metrics. KLD is not defined in this path.
2. **CVAE objective replay:** provide expert actions and use the
   action-conditioned VAE path without gradients. Use a fixed episode order and
   three declared random seeds by default. Reload the checkpoint before every
   seed and report mean and standard deviation.

The CVAE pass is labelled `recomputed objective at checkpoint`. It is not the
loss observed while that checkpoint was originally trained.

## Inputs and Data Flow

The command accepts:

- one single-floor button-press dataset root;
- an explicit episode list;
- one or more compatible checkpoint paths;
- device, batch size, CVAE seeds, and output directory.

It performs these steps:

1. Validate that all checkpoints use the current eight-state, L515 image, and
   seven-action schema.
2. Record checkpoint paths/steps, dataset path, episode IDs, configuration, and
   evaluation seeds in `evaluation_manifest.json`.
3. For each checkpoint, reuse the existing deployment/expert evaluation.
4. Reload that checkpoint and run CVAE objective replay on the identical
   episode subset.
5. Read normalization mapping and saved processor statistics.
6. Aggregate all checkpoint rows and generate the static visual report.

A checkpoint requiring `observation.environment_state` fails validation with a
clear out-of-scope message.

## Visual Output

Version one produces a self-contained `report.html` and simple PNG charts using
the Pillow path already used by the evaluator. It adds no plotting framework.

The report contains:

1. **Availability/configuration table** — `use_vae`, `kl_weight`, LRs,
   normalization mapping, and exact/recomputed/inferred/unavailable labels.
2. **Loss chart** — one checkpoint per x-axis position, with L1 and weighted KLD
   shown as stacked components, total shown explicitly, and KLD also available
   as an unweighted value in the table. The title includes `recomputed`, the
   episode set, and seed count.
3. **Expert comparison** — the existing normalized L1, translation error,
   rotation error, ADE/FDE, XYZ RMSE, per-action physical MAE, gripper metrics,
   and horizon plot shown side by side across checkpoints.
4. **Normalization table** — mapping and saved image/state/action statistics.
   It is descriptive; version one does not add distribution-drift or latent
   diagnostics.
5. **Warnings** — missing historical loss, unknown train membership, gripper
   imbalance, failed checkpoints, and teacher-forced evaluation limitations.

Commanded cumulative delta XYZ must be labelled as a commanded-delta trajectory,
not a measured absolute end-effector trajectory. Offline agreement must not be
presented as closed-loop success.

## Output Contract

```text
<output_dir>/
  evaluation_manifest.json
  checkpoint_metrics.csv
  loss_replay.csv
  normalization_summary.json
  checkpoints/<checkpoint_id>/
    per_episode_metrics.csv
    summary_metrics.json
    horizon_translation_error.csv
    horizon_error.png
  loss_components.png
  checkpoint_comparison.png
  report.html
```

Existing per-checkpoint filenames are retained. The top-level files only add
cross-checkpoint and loss summaries.

## Error Handling

- Fail before inference on incompatible feature names, shapes, action order,
  missing processor statistics, invalid episode IDs, or non-finite inputs.
- Do not mix checkpoints that require different schemas.
- A failed checkpoint is recorded in the report while independent compatible
  checkpoints may still finish; the command returns nonzero if any requested
  checkpoint fails.
- A replay result must satisfy
  `total ~= L1 + kl_weight * KLD` within floating-point tolerance. Violation is
  treated as an implementation error rather than plotted.
- Existing output directories are not silently overwritten.

## Testing

Focused unit tests cover:

- ACT L1 padding mask, KLD, weighted KLD, and formula identity;
- exact/recomputed/inferred/unavailable classification;
- extraction of configuration and normalization statistics;
- rejection of multi-floor/incompatible checkpoints;
- loss-chart/report generation when historical loss is unavailable.

Integration tests with fake policies and datasets prove that every checkpoint
uses the same episodes, deployment inference is deterministic, CVAE seeds are
repeatable, and the recomputed loss matches ACT's forward loss on the same
synthetic batch.

A final smoke test loads a real single-floor checkpoint and a small episode
subset, produces all output files, and verifies finite values. Automated tests
never access robot hardware.

## Acceptance Criteria

- The 29-episode and 50-episode single-floor checkpoints can be evaluated on one
  explicit episode subset and compared in one report.
- The report marks historical training L1, KLD, total loss, and gradient norm as
  unavailable.
- Recomputed L1, KLD, weighted KLD, and total are shown per checkpoint and
  satisfy the ACT loss formula.
- Existing expert-comparison metrics remain available and are visually
  comparable across checkpoints.
- Configuration and saved normalization information are visible without being
  mistaken for measured training curves.
- No training source file is modified, no JSONL logger is added, multi-floor
  checkpoints remain out of scope, and no new dependency is introduced.

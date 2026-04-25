# ThaiSpoof Config Cleanup Design

## Goal

Make the ThaiSpoof experiment pipeline easier to run on the current MacBook Air M4 while keeping the same code configurable for a higher-performance computer or Colab GPU. The next change should reduce fragile command-line repetition, make the downloaded local dataset layout explicit, and avoid expensive repeated feature extraction.

## Current Context

The workspace has two downloaded data roots:

- `genuine/` contains bona fide Thai speech.
- `Corpus-Spoof-VAJA/` contains spoofed Thai speech with `Train/` and `Test/` audio folders.

The existing `ThaiSpoof/project` package already provides dataset discovery, balanced splitting, LFCC/MFCC extraction, a small CNN baseline, a lightweight ResNet improvement model, metrics, and unit tests. A probe run from the workspace root detected 4,583 genuine files and 4,583 spoof files.

## Proposed Approach

Use the existing Python package and add a small configuration layer instead of replacing the pipeline.

1. Add named experiment presets for common hardware profiles.
   - `smoke`: tiny run for verifying paths, feature extraction, and training wiring.
   - `mac_small`: safe default for MacBook Air M4 with 16 GB RAM.
   - `balanced_medium`: larger local run for a machine with more memory/CPU.
   - `high_perf`: higher sample counts, larger frame length, and larger batch size for a strong workstation or GPU runtime.

2. Add optional external config loading.
   - A config file can override preset values without editing source code.
   - Command-line arguments still override both preset and config file values.
   - Keep the format simple and standard-library friendly: JSON.

3. Add dataset layout helpers.
   - Support the current local layout directly from the workspace root.
   - Keep recursive class inference for renamed or nested data folders.
   - Add a dry summary command so the user can confirm class counts before training.

4. Add feature-cache reuse.
   - If the expected feature pickle groups already exist for the requested split counts and feature type, reuse them.
   - Provide a force flag to rebuild features when needed.

5. Update docs with two command paths.
   - First: `smoke` on the current Mac.
   - Later: `high_perf` using a JSON config on a stronger computer.

## Data Flow

The CLI will resolve settings in this order:

1. Built-in defaults.
2. Named preset.
3. JSON config file, if supplied.
4. Explicit command-line arguments.

Then it will:

1. Collect audio from `data_root`.
2. Create or load a manifest.
3. Extract or reuse cached features.
4. Train and evaluate the selected model.
5. Write metrics and summary files under the chosen output directory.

## Example Config

```json
{
  "preset": "high_perf",
  "data_root": ".",
  "out_dir": "ThaiSpoof/runs/lfcc_high_perf",
  "feature": "lfcc",
  "model": "resnet_lite",
  "train_genuine": 3000,
  "train_spoof": 3000,
  "test_genuine": 1500,
  "test_spoof": 1500,
  "dim_x": 256,
  "batch_size": 32,
  "epochs": 40,
  "patience": 8
}
```

## Error Handling

- Missing data roots should fail with a clear `FileNotFoundError`.
- Empty genuine or spoof pools should fail before feature extraction begins.
- Invalid preset names or config keys should fail before doing expensive work.
- Cache reuse should only happen when the requested feature group files are present; otherwise extraction proceeds normally.

## Testing

Add focused tests for:

- Preset resolution and command-line override behavior.
- JSON config loading.
- Current local dataset layout detection.
- Feature-cache decision logic.

Keep tests fast and filesystem-local. Do not require TensorFlow training in unit tests.

## Scope

This change will not train the final report models yet. After the cleanup passes tests, run a tiny `smoke` experiment first. Larger baseline and improvement experiments can follow using the new presets/config file.

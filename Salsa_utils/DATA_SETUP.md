# InterGen data setup (InterHuman)

InterGen expects a single **DATA_ROOT** folder with this layout:

```
DATA_ROOT/
├── train.txt          # Motion names, one per line (e.g. 3433)
├── val.txt
├── test.txt
├── ignore_list.txt    # Optional; motion names to skip (can be empty)
├── motions_processed/
│   ├── person1/
│   │   └── <name>.npy
│   └── person2/
│       └── <name>.npy
└── annots/
    └── <name>.txt     # Three sentences per file
```

## Your data: `Baselines/InterHuman_DATASET`

Your dataset already has the right **content**:

- `motions_processed/person1/`, `person2/` with `.npy` files
- `annots/` with `.txt` files (three sentences each)
- `split/train.txt`, `split/val.txt`, `split/test.txt` (motion names, one per line)

InterGen expects `train.txt`, `val.txt`, `test.txt` **in the DATA_ROOT root**, not in a `split/` subfolder.

### What was done

1. **Symlinks** in `InterHuman_DATASET/` so the loader finds the splits:
   - `train.txt` → `split/train.txt`
   - `val.txt` → `split/val.txt`
   - `test.txt` → `split/test.txt`

2. **`ignore_list.txt`** created in `InterHuman_DATASET/` (optional file; can stay empty).

3. **Config** in `configs/datasets.yaml` set to:
   - `DATA_ROOT: ../InterHuman_DATASET`  
   (relative to `Baselines/Salsa_InterGen` when you run the app or training from there.)

### If you prefer a different DATA_ROOT

- Use an **absolute path** in `configs/datasets.yaml`, e.g.  
  `DATA_ROOT: /path/to/Baselines/InterHuman_DATASET`
- Or keep data under `Salsa_InterGen/data/` and name it e.g. `data/interhuman_processed` with the same layout (including `train.txt`, `val.txt`, `test.txt` in that folder).

### Encoding of annot files

InterGen opens annot `*.txt` files with the default encoding (UTF-8). If any file contains non-UTF-8 bytes (e.g. Windows-1252 smart quotes, byte `0xa1`), you get `UnicodeDecodeError`. Re-save those files as UTF-8 (or run a script that reads with `errors='replace'` or `latin-1` and writes as UTF-8). Four such files were fixed: `4642.txt`, `5834.txt`, `5834(1).txt`, `4642(1).txt`.

### Line endings in split files

InterGen’s dataset checks motion names with **Unix line endings** (`\n`). If `split/train.txt`, `split/val.txt`, and `split/test.txt` have Windows line endings (`\r\n`), no motions will match and the dataset will be empty. Ensure these files use Unix line endings (e.g. run a script to replace `\r\n` with `\n`). This has been done once for the current split files.

### Normalization (unchanged)

`global_mean.npy` and `global_std.npy` are still loaded from **`Salsa_InterGen/data/`** (used by the model). They are separate from DATA_ROOT.

# ASAG Model Artifacts

This directory is the runtime home for the MindSpore ASAG short-answer scoring model.

Expected artifacts:
- `asag_mohler_best.ckpt`
- `asag_mohler.mindir`
- `tokenizer/vocab.json`
- `results.json`

Current status:
- `tokenizer/vocab.json` and `results.json` are staged here.
- The trained checkpoint and MindIR generated in `msmodels/workspace/asag/outputs/asag_mohler/` are currently root-owned and unreadable to the service user.
- Copy or chmod the trained artifacts into this directory before enabling production ASAG scoring.

Recommended env:
```bash
export ASAG_MODEL_DIR="$PWD/models/asag"
export ASAG_CKPT_PATH="$ASAG_MODEL_DIR/asag_mohler_best.ckpt"
export ASAG_MINDIR_PATH="$ASAG_MODEL_DIR/asag_mohler.mindir"
export ASAG_VOCAB_PATH="$ASAG_MODEL_DIR/tokenizer/vocab.json"
export ASAG_RESULTS_PATH="$ASAG_MODEL_DIR/results.json"
```

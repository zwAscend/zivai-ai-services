Active DKT artifacts for the AI service.

Use these artifacts for the cloud/runtime integration:
- dkt_lstm_cloud.ckpt
- skill_map_v1.json
- model_meta.json

Notes:
- `dkt_lstm_cloud.ckpt` is the active cloud checkpoint referenced by `README_DKT_MODEL.md`.
- `dkt_lstm_edge.mindir` is kept here for later edge conversion/testing, not as the primary backend runtime artifact.
- The older root-level `dkt_model*.mindir` files are not used by the live DKT service.

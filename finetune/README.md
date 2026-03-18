# Fine-tuning (LoRA) – MindNLP

The old standalone-ASAG training export flow has been removed together with the
deprecated local `questions`, `rubric_items`, and `submissions` tables.

Current state:
- `finetune/build_train_jsonl.py` has been removed
- the future training/export path must be rebuilt from shared `lms.*` and
  `ai.*` tables
- LoRA training scripts should only be used after the shared-schema grading
  pipeline is in place

The remaining training script is:

```bash
python finetune/train_lora_qwen25_15b.py --train train.jsonl --model_id Qwen/Qwen2.5-1.5B-Instruct --output_dir lora_out
```

But `train.jsonl` must now come from a new shared-schema exporter that does not
yet exist.

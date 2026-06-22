# Artifacts

Required for full reproduction:

- `t539_selector/cross_encoder_model/`: query/span selector model.
- `t488_reader/qwen3_consumer_adapter_latest.pt`: layer 0-7 no-packet reader adapter.
- `rmt/qwen3_rmt_joint_memory_latest.pt`: recurrent memory checkpoint used by the original evaluator.
- `sae/topk_sae_latest.pt`: sparse autoencoder checkpoint used by the original evaluator.
- `selector_materialization/feature_manifest.json`: structured selector feature manifest.
- `longmem_inputs/*.jsonl`: materialized LongMemEval dev rows for the bundled dev reproduction.

Optional:

- `t576_reader/qwen3_consumer_adapter_latest.pt`: narrow numeric repair reader used only for exact T613 reproduction.

External:

- Qwen3-4B base model.
- A Python/CUDA environment compatible with PyTorch and Transformers.


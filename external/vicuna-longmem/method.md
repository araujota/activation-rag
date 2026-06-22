# Method

The system turns mechanistic memory signals into a practical selector.

1. Prior conversation spans are materialized as selector rows.
2. CAA/SAE and ordinary text metadata are attached as structured features.
3. The T539 selector scores query/span pairs for actual answer evidence.
4. The top four spans are expanded to sentence envelopes.
5. The reader sees raw selected memory text and the final question.
6. The T488 layer 0-7 adapter helps Qwen3-4B consume that selected memory.
7. The T613 policy post-processes the generated answer only in narrow audited cases.

CAA means causal activation/attribution analysis: checks that a span is not merely topically related, but has causal evidence for answer behavior.

SAE means sparse autoencoder: a tool that decomposes hidden activations into sparse features. In this release, SAE labels are not shown to the answer model. SAE-derived values are used as structured features and audit signals.

RMT means recurrent memory transformer: a learned memory mechanism that writes compact state across prior spans.

The important design choice is that the answer model receives selected raw text, not summaries or feature-label prose. This keeps the mechanism close to retrieval while using mechanistic features to improve what gets retrieved.


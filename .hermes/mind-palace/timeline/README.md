# Timeline — WuBuText AI Build Order

## Phase Order

```
Week 1:  Embedding Grafting (Phase 1)
  ├── Step 1.1: GGUF reader + extract embeddings
  ├── Step 1.2: Analyze distribution, determine R
  ├── Step 1.3: Poincaré mapping
  └── Step 1.4: Verify + test on baseline

Week 2:  Attention Port (Phase 2)
  ├── Step 2.1: Standard Gated DeltaNet in C
  ├── Step 2.2: Hyperbolic gyration variant
  ├── Step 2.3: GQA full attention
  └── Step 2.4: Test convergence

Week 3:  Training Loop (Phase 3)
  ├── Step 3.1: Data pipeline
  ├── Step 3.2: Loss + MTP
  ├── Step 3.3: WuBu optimizer
  ├── Step 3.4: Training config
  ├── Step 3.5: CUDA kernels
  └── Step 3.6: Checkpointing

Week 4:  MoE Port (Phase 4)
  ├── Step 4.1: Standard MoE router
  ├── Step 4.2: Hyperbolic distance routing
  ├── Step 4.3: Nested geometry tree
  └── Step 4.4: Shared expert

Week 5:  Integration + Training
  ├── Full model integration
  ├── Hyperparameter tuning
  └── First end-to-end training run

Week 6:  Validation + Vision (Phase 5)
  ├── Benchmarking
  ├── WuBuVision encoder port
  └── Report results
```

## Progress Tracking

Each step marked [ ] = not started, [/] = in progress, [X] = done.

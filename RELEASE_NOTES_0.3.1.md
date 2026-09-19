# RunNX 0.3.1 Release Notes

*Prepared on September 19, 2026*

RunNX 0.3.1 is a correctness and release-hardening patch release. It focuses on malformed-model safety, ONNX semantic compatibility, deterministic graph execution, trustworthy diagnostics, and a reproducible publication process.

## Highlights

### Safer model loading and tensor shapes

- Negative ONNX dimensions are rejected before conversion to `usize`.
- Shape products and model-controlled allocation sizes use checked arithmetic.
- Conv, MaxPool, Pad, Reshape, Split, Gather, ConstantOfShape, and Resize reject malformed or unrepresentable parameters with actionable errors.
- Graph validation now rejects cycles, duplicate tensor producers, duplicate initializers, and ambiguous optional-input holes.

### Corrected ONNX behavior

- Conv defaults to zero padding when `pads` is absent.
- Softmax honors any valid axis, including axis 0 on matrices.
- Gather accepts valid negative indices and rejects out-of-range, fractional, NaN, or infinite indices.
- Slice supports normalized negative axes, negative steps, and valid empty ranges.
- Split preserves zero-length outputs so runtime output counts remain correct.
- MaxPool emits negative infinity for windows containing only padding.
- ReduceMean honors axes, negative axes, `keepdims`, and axes supplied as an input.
- ReLU and Sigmoid handle IEEE-754 NaN and infinity values.
- Conv, Gather, and BatchNormalization work with non-standard ndarray layouts.

### More reliable formats, CLI, and diagnostics

- ONNX serialization preserves model versions and supported scalar/list/string attributes, including scalar tensor values used by ConstantOfShape.
- `Model::run_with_stats` returns measured operation counts, timings, and memory usage.
- `runnx-runner --version` is implemented, and CLI documentation now matches available flags.
- Unsupported operator paths fail explicitly instead of returning plausible but incorrect tensors.

### Formal verification and release automation

- Nine previously dormant property tests are active and pass.
- Why3 prover selection uses the installed Alt-Ergo shortcut rather than a pinned prover version.
- Formal CI no longer masks property-test or proof failures.
- Normal builds use checked-in protobuf bindings; `protoc` is only needed with `RUNNX_REGENERATE_ONNX_PROTO=1`.
- The vulnerable Crossbeam transitive dependency is upgraded, warned build/dev dependencies are refreshed, and the unused `imageproc` dependency is removed; `cargo audit` is clean.
- Tagged releases verify tag/version/changelog consistency, stable-branch ancestry, formatting, Clippy, tests, docs, release builds, and package dry runs before publication.

## Compatibility

This is a SemVer patch release. Existing supported APIs remain available. Some malformed or unsupported operations that previously appeared to succeed now return errors; this prevents silent wrong inference results.

## Known Limitations

- Tensors are internally represented as `f32`; general ONNX data-type preservation is not implemented.
- Conv supports 2D NCHW inference with `group = 1`, unit dilation, and explicit
	or zero padding. MaxPool supports 2D NCHW inference with unit dilation, floor
	output sizing, row-major storage order, and explicit or zero padding.
- Resize supports nearest-neighbor spatial scaling for 4D NCHW tensors with batch and channel scales fixed at 1.
- Pad supports constant mode only, and Cast supports only a float32 target.
- Interior omitted optional ONNX inputs are rejected because the runtime cannot yet preserve positional holes.
- Upsample and NonMaxSuppression are recognized but intentionally return unsupported-operation errors.
- Full end-to-end compatibility with arbitrary YOLO exports is not claimed.

## Upgrade

```toml
[dependencies]
runnx = "0.3.1"
```

No source migration is expected for callers already using the supported operator and format subset.

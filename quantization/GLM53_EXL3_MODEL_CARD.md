---
base_model: zai-org/GLM-5.3
language:
- en
- zh
library_name: transformers
license: mit
pipeline_tag: text-generation
tags:
- glm
- mixture-of-experts
- exl3
- 4-bit
---

# GLM-5.3 EXL3 K4 v1

This repository contains the calibrated EXL3 K4 model
`wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1`, derived from
[`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3) for GLMRT. The
three dense decoder layers, attention, routers, shared experts, embeddings,
head, and native MTP layer retain their source tensors. Only the 256 routed
experts in base decoder layers 3 through 77 are replaced by EXL3 K4 MCG
`trellis/suh/svh/mcg` tensors.

The artifact uses one resident expert representation on each of four Spark TP
ranks. Every rank retains its 512-wide intermediate slice of every expert;
the runtime does not keep source and EXL3 expert copies resident together.

## Quantization

- Source: `zai-org/GLM-5.3`, revision `935644c05e76fc198714f4cca449fd8b970ff6d7`
- Source expert format: FP8 E4M3 with block-128x128 scales
- Format: EXL3, 4 physical trellis bits per routed-expert weight
- Codebook: MCG
- Calibration: 1,082,141 GLM-5.3 tokenizer tokens across 1,441 examples
- Sparse coverage: natural top-8 routes with the complete calibrated projection
  inventory retained as separately verifiable quantization evidence
- Quantizer: the content-pinned GLMRT GPTQModel fork

The complete immutable plan, projection evidence, retained-native proof, and
artifact manifest are maintained by the GLMRT project rather than shipped as
large private recovery files in this standard model repository. The published
`quantize_config.json` does retain the complete EXL3 tensor-storage map needed
to interpret every routed projection, together with the quantizer's calibrated
error-ledger provenance. The main `config.json` embeds only the four EXL3
discovery fields (`quant_method`, `format`, `checkpoint_format`, and `bits`),
so the complete calibration and storage metadata is not duplicated there.
Only redundant top-level GPTQModel execution controls
such as active offload and CUDA-placement settings are excluded; the
content-bound calibration ledger itself, including its recorded source and
execution provenance, is retained unchanged.

## Runtime

The artifact is intended for GLMRT's generated SparkInfer SM121 EXL3 K4 TP4
path. GLMRT qualifies both native MTP and `incoai/GLM-5.3-DFlash2`; DFlash2 is
the production default. Its adaptive verifier proposes as many as seven draft
tokens and chooses each verification width from a route-calibrated target-cost
profile. A fixed K5 arm is retained as the performance reference for the
adaptive policy. Tool quality is qualified independently from that throughput
choice rather than used to tune a speculative width.
Generic Transformers metadata is included, but compatibility with other EXL3
runtimes has not been claimed.

## Qualification

GLMRT_PUBLICATION_RESULTS_PENDING

Before publication this marker is replaced mechanically from the signed final
structural, quantizer, and serving reports. The publication builder rejects a
card that retains the marker or does not cite the exact serving-report hash.

## License and attribution

This derivative follows the source model's MIT license. See `LICENSE` and the
original [`GLM-5.3` model card](https://huggingface.co/zai-org/GLM-5.3) for
upstream architecture, intended-use, and limitation details.

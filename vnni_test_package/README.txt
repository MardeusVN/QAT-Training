Piper VITS voice FP32 vs INT8 benchmark package
================================================

Files:
  voice.fp32.onnx      - baseline FP32 model (70.6 MB)
  voice.int8.onnx      - quantized INT8 model (46.4 MB), flow + final
                          output layer excluded from quantization, per-
                          channel weights, QOperator format
  test_sentences.json  - 5 sample sentences with pre-computed phoneme_ids
                          (141-325 phonemes, no text-to-phoneme step needed)
  bench_vnni.py         - the benchmark script

Setup on the target machine (only needs Python 3.9+):
  pip install onnxruntime numpy

Run:
  python bench_vnni.py

What it does:
  Runs both models at intra_op_num_threads=2 then =1 (the "1 core / 2
  threads" edge budget), 8 timed runs per sentence after 2 warmup runs,
  and prints the average INT8-vs-FP32 speedup at each thread count.

Why this matters:
  On the original dev machine (Intel i5-10400F, Comet Lake -- no AVX-VNNI),
  INT8 was only ~1.05-1.08x faster than FP32. Comet Lake predates Intel's
  AVX-VNNI instruction set (introduced with Alder Lake in 2021), so ONNX
  Runtime's quantized kernels there get essentially no dedicated INT8
  compute acceleration -- the small speedup measured is mostly a memory-
  bandwidth effect from the ~4x smaller weight tensors, not faster compute.

  Machines with AVX-VNNI (Intel 12th gen "Alder Lake" and newer -- includes
  Alder Lake-N / N100 / N150, and Raptor Lake i5/i7/i9-13xxx/14xxx; AMD
  Zen4 and newer) have dedicated INT8 dot-product instructions ONNX
  Runtime can dispatch to automatically, which typically gives INT8 models
  meaningfully bigger real speedups than what was measured here. Re-running
  this benchmark on such a machine is the way to find out how much bigger,
  for this specific model.

To confirm AVX-VNNI is actually available on the target machine:
  Windows: check the exact CPU model against Intel ARK / AMD's spec page.
  Linux:   cat /proc/cpuinfo | grep -o 'avx_vnni' | head -1

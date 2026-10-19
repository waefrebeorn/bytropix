# solidcv reimplementation — module registry

Goal: reimplement the useful surface of
[solidsf-inc/solidcv](https://github.com/solidsf-inc/solidcv) in our own C11,
one tested module at a time. Nothing is imported; the crate is a specification,
not a dependency.

Statuses are strict: **done** means a module exists here and a test runs in
`make test_all`. **partial** means a real subset exists and the missing part is
named. **queued** means nothing exists. **rejected** means we decided not to.

| solidcv module | Ours | Status | Evidence / what is missing |
|---|---|---|---|
| `core` (Mat, Scalar, Rect, SVD, solve) | `wubu_cv` (`wubu_img`) | **partial** | alloc/clone/fill/at tested; SVD, linear solve and multi-channel types absent |
| `imgproc` (filters, edges, Otsu, morphology, contours) | `wubu_cv` | **partial** | integral image, O(1) box mean/var, fused 3×3 Sobel, Otsu, 3×3 Gaussian, **Canny (NMS + hysteresis)** and **8-connected contour boxes** tested; morphology, GrabCut and full contour point tracing queued |
| `features2d` (ORB, BFMatcher) | `wubu_features` | **partial** | FAST-9, 256-bit BRIEF, Hamming tested; ORB pyramid, Harris scoring and a full matcher queued |
| `calib3d` (StereoSGBM, PnP, chessboard, ArUco) | — | **queued** | needs a calibrated camera pipeline; not on the eyes' critical path yet |
| `video` (Lucas–Kanade, Farneback) | — | **queued** | optical flow is the next rung after patch differencing |
| `pixelean` (bitplanes, D4, SE(2) jets) | `wubu_eye` + `wubu_uplift` | **partial** | XOR+popcount carrier and D4 canonical hash tested; bitplane partition and SE(2) jets queued |
| `voxelean` (octree, Betti, wavelets) | — | **queued** | no 3D volume pipeline to attach it to |
| `worldtube` ((3+1)D, CFL cones) | — | **queued** | requires metric camera calibration |
| `realitycomb` (static/moving separation) | — | **queued** | depends on flow (video) |
| `hardware` (NEON/Metal/PTX) | CPU C11 | **partial (deliberate)** | portable integer/SIMD-friendly code; no NEON/Metal; CUDA lives in the engine, not here |
| `mechinterp` (SAE matcher, patch pruner, GlassBox) | `wubu_eye` | **partial** | patch-token pruner tested (26 checks); 256-bit SAE atom matcher and the 64-D signature queued |
| `uplift` (mass-conserving zoom, burst SR, foveated ROIs) | `wubu_uplift` | **partial** | 2× uplift and 2× box down tested as an exact right-inverse (round-trip equality) plus D4-invariant hash; burst SR and foveated ROI selection queued |
| `cloud_ws` (metered WebSocket) | — | **rejected** | a paid external service; the run policy is third-party on our kernel, never the reverse |

## Tests

- `test_eye` — 26 checks: patch masks, XOR/popcount, tolerance vs threshold, partial patches.
- `test_cv` — **39 checks**: image core, integral + O(1) box stats, Sobel sign/orientation,
  Otsu on a bimodal image, Gaussian edge clamp, FAST-9 on a square, BRIEF determinism,
  Hamming, uplift mass conservation, `down2x(uplift2x(x)) == x`, D4 rotation invariance,
  Canny on one rectangle and two squares, contour bbox origin/size and thinness.

## Next rungs, in order

1. **Morphology + full contour tracing** (imgproc) — dilate/erode, then point lists rather than boxes.
2. **ORB pyramid + matcher** (features2d) — track a widget across frames.
3. **Lucas–Kanade** (video) — motion, which also unlocks `realitycomb`.
4. **SAE binary matcher** (mechinterp) — 256-bit Hamming argmin is already half-written by `wubu_hamming256`.
5. **Foveated ROI selection** (uplift) — the decision plane already knows what a token costs.

## Honest note on the source

The crate is days old with a small star count, and its benchmark table was not
reproduced here. We are reimplementing the *ideas it describes*, and we test our
own arithmetic; we do not inherit its performance claims.

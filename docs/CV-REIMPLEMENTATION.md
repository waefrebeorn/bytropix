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
| `imgproc` (filters, edges, Otsu, morphology, contours) | `wubu_cv` | **partial** | integral image, O(1) box mean/var, fused 3×3 Sobel, Otsu, 3×3 Gaussian, Canny (NMS + hysteresis), 8-connected contour boxes, **ordered contour point lists**, **3×3 dilate/erode**, **foveated ROI tiles**; GrabCut queued |
| `features2d` (ORB, BFMatcher) | `wubu_features` | **done (core)** | FAST-9, 256-bit BRIEF, Hamming, 2-level ORB pyramid, ratio-test matcher, **Harris response** and **NMS dedupe** tested; multi-scale descriptor fusion queued |
| `calib3d` (StereoSGBM, PnP, chessboard, ArUco) | — | **queued** | needs a calibrated camera pipeline; not on the eyes' critical path yet |
| `video` (Lucas–Kanade, Farneback) | `wubu_flow` | **partial** | single-scale LK (2,3 exact), coarse-to-fine pyramid LK (6,7 exact), dense block-matching flow (36/36 interior blocks) sub-pixel parabolic refinement (0.125/0.109 px) and **dense Horn-Schunck** (0.206/0.151 px on a (2,3) shift) tested; Farneback queued |
| `pixelean` (bitplanes, D4, SE(2) jets) | `wubu_eye` + `wubu_uplift` | **partial** | XOR+popcount carrier and D4 canonical hash tested; bitplane partition and SE(2) jets queued |
| `voxelean` (octree, Betti, wavelets) | — | **queued** | no 3D volume pipeline to attach it to |
| `worldtube` ((3+1)D, CFL cones) | — | **queued** | requires metric camera calibration |
| `realitycomb` (static/moving separation) | — | **queued** | depends on flow (video) |
| `hardware` (NEON/Metal/PTX) | CPU C11 | **partial (deliberate)** | portable integer/SIMD-friendly code; no NEON/Metal; CUDA lives in the engine, not here |
| `mechinterp` (SAE matcher, patch pruner, GlassBox) | `wubu_eye` + `wubu_sae` + `wubu_glass` | **done (core)** | patch-token pruner (26 checks), 256-bit Hamming argmin, and the **64-D GlassBox signature + D4 canonical hash** tested; worldtube coordinates reserved rather than invented |
| `uplift` (mass-conserving zoom, burst SR, foveated ROIs) | `wubu_uplift` + `wubu_cv` | **partial** | 2× uplift / 2× box down round-trip and D4 hash tested; **foveated ROI tile selection** and **multi-frame burst SR** tested (RMSE 0.885 vs 7.642 for naive 2x) |
| `cloud_ws` (metered WebSocket) | — | **rejected** | a paid external service; the run policy is third-party on our kernel, never the reverse |

## Tests

- `test_eye` — 26 checks: patch masks, XOR/popcount, tolerance vs threshold, partial patches.
- `test_look` — 9 checks: two frames -> ranked changed ROIs + reuse ratio (93.8% reuse for a 32x32 change) and a token-budgeted variant that stops at the paid budget.
- `test_cv` — **147 checks**: image core, integral + O(1) box stats, Sobel sign/orientation,
  Otsu on a bimodal image, Gaussian edge clamp, FAST-9 on a square, BRIEF determinism,
  Hamming, uplift mass conservation, `down2x(uplift2x(x)) == x`, D4 rotation invariance,
  Canny on one rectangle and two squares, contour bbox origin/size and thinness,
  ORB pyramid + ratio-test matcher, morphology cores, LK flow on a known shift,
  SAE argmin/signature, foveated ROI selection, Harris scoring, dedupe, pyramid LK,
  ordered contour tracing (244 points, closed loop), burst super-resolution,
  block-matching flow, sub-pixel refinement, dense Horn-Schunck, checkerboard
  corners and grid ordering.

## Next rungs, in order

1. **Farneback proper** (video) — polynomial-expansion dense flow; the shortcut is documented as rejected, HS covers the dense need meanwhile.
2. **calib3d** — chessboard corners, then PnP.
3. **GrabCut** (imgproc) — interactive segmentation.

## Honest note on the source

The crate is days old with a small star count, and its benchmark table was not
reproduced here. We are reimplementing the *ideas it describes*, and we test our
own arithmetic; we do not inherit its performance claims.

## Rejected approaches (do not retry)

- **Farneback via linear moments.** Approximating the polynomial's linear
  coefficient with `b = <Ix·I>` and solving `d = -0.5 A⁻¹ (b2 - b1)` produced
  mean errors of 475 px in x and 351 px in y on a known (2,3) shift — the
  shortcut drops the intensity-centering that the real expansion gets from
  normalized convolution. Reverted rather than shipped. The correct route is a
  6×6 normalized-convolution fit over the basis `[1, x, y, x², xy, y²]`
  followed by the displacement solve; anything less is guesswork.


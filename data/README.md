# Benchmark data

All files in `boileau2015/` come from the data supplement of

> E. Boileau, P. Nithiarasu, P. J. Blanco, L. O. Müller, F. E. Fossan, L. R. Hellevik,
> W. P. Donders, W. Huberts, M. Willemet, J. Alastruey. A benchmark study of numerical
> schemes for one-dimensional arterial blood flow modelling. *Int. J. Numer. Meth.
> Biomed. Engng.* 31(10):e02732, 2015. doi:10.1002/cnm.2732

distributed on figshare (doi:10.6084/m9.figshare.1586944) under the
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) public-domain
dedication. The 3D reference solutions (`3D.txt`) were computed by Xiao, Alastruey
and Figueroa:

> N. Xiao, J. Alastruey, C. A. Figueroa. A systematic comparison between 1-D and 3-D
> hemodynamics in compliant arterial models. *Int. J. Numer. Meth. Biomed. Engng.*
> 30(2):204–231, 2014.

The six 1D solutions are those of the codes DCG, LCG, FEM, FVM, McC and STM.

| Directory | Supplement source | Contents |
|---|---|---|
| `single_pulse/` | `Benchmark1_SinglePulseModel/NumData/<M>_{inviscid,viscous}.txt` | Unchanged copies. Columns: x [m], then p [kPa] at t = 0.1, 0.3, …, 1.5 s. |
| `carotid/` | `Benchmark2_CommonCarotidArtery/NumData/` | `<M>.txt`: unchanged contiguous row subsets, 8.805 ≤ t ≤ 10.005 s. `3D.txt`: one cycle, line endings converted from CRLF to LF. |
| `aorta/` | `Benchmark3_UpperThoracicAorta/NumData/` | `<M>.txt`: unchanged contiguous row subsets, 18.098 ≤ t ≤ 19.153 s. `3D.txt`: one cycle, CRLF to LF. |

The carotid and aorta files have the columns t [s], p [kPa], Q [ml/s], radius change
from diastole [mm] and p_inlet − p_outlet [kPa]. The rows kept cover the cycle used
for the comparison (see the phase rule in `src/benchmark.jl`). The inflow Fourier
series of the supplement's `inflow.txt` files and the parameters of the paper's
Tables II, III and V are transcribed in `experiments/`.

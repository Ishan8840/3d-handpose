| Pipeline | Abs MPJPE | Wrist | Fingertips | Coverage | P90 | P95 | Depth MAE | >100 mm joints |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WiLoR stereo (preserved) | 22.892 | 21.384 | 30.962 | 58.12% | 40.931 | 55.835 | 17.611 | 1.09% |
| Existing MANO + 50 ms temporal | 24.267 | 29.940 | 26.983 | 67.33% | 35.636 | 47.669 | 18.386 | 2.01% |
| A: controlled MANO refit | 25.950 | 32.842 | 28.019 | 67.69% | 36.873 | 49.279 | 18.784 | 1.72% |
| B: mask visibility | 25.950 | 32.839 | 28.014 | 67.69% | 36.893 | 49.279 | 18.795 | 1.72% |
| C: estimated object + collision | 25.956 | 32.839 | 28.026 | 67.69% | 36.887 | 49.201 | 18.793 | 1.72% |
| E: estimated object + contact | 25.950 | 32.835 | 28.007 | 67.69% | 36.887 | 49.201 | 18.790 | 1.72% |
| D: reference pose, same acceptance | 25.946 | 32.835 | 27.998 | 67.69% | 36.881 | 49.279 | 18.776 | 1.72% |
| Oracle: collision, all selected | 25.960 | 32.832 | 28.028 | 67.69% | 36.893 | 49.279 | 18.768 | 1.72% |
| Oracle: contact, all selected | 25.929 | 32.807 | 27.990 | 67.69% | 36.824 | 49.007 | 18.731 | 1.72% |

# Locked NARMA benchmark protocol

Configuration hash: `917b7fdf6f8ed15ff9f2a8820a07ed6e716584622a879218c478b4daf240ef7e`

Each independently reset sequence consumes `u[t]` before predicting `y[t+1]`. MSE and NRMSE exclude the first 200 samples. NRMSE is `sqrt(MSE / population_variance(target))`.

The final test streams are constructed only by the final phase, after the selected configurations have been locked.

## Task definitions

### narma5_fujii_nakajima

`y[t+1] = 0.3 y[t] + 0.05 y[t] sum(y[t-i], i=0..4) + 1.5 u[t-4] u[t] + 0.1`

Each proposal has `u[t] iid ~ Uniform(0, 0.2)` from NumPy PCG64. Source: [Fujii and Nakajima (2017), Eq. 18 and Appendix A.2](https://doi.org/10.1103/PhysRevApplied.8.024030).

### narma10_atiya_parlos

`y[t+1] = 0.3 y[t] + 0.05 y[t] sum(y[t-i], i=0..9) + 1.5 u[t-9] u[t] + 0.1`

Each proposal has `u[t] iid ~ Uniform(0, 0.5)` from NumPy PCG64. Source: [Atiya and Parlos (2000), Eq. 86](https://doi.org/10.1109/72.846741).

### narma20_rodan_tino

`y[t+1] = tanh(0.3 y[t] + 0.05 y[t] sum(y[t-i], i=0..19) + 1.5 u[t-19] u[t] + 0.01)`

Each proposal has `u[t] iid ~ Uniform(0, 0.5)` from NumPy PCG64. Source: [Rodan and Tino (2011), Eq. 6](https://doi.org/10.1109/TNN.2010.2089641).

### narma30_schrauwen

`y[t+1] = 0.2 y[t] + 0.04 y[t] sum(y[t-i], i=0..29) + 1.5 u[t-29] u[t] + 0.001`

Each proposal has `u[t] iid ~ Uniform(0, 0.5)` from NumPy PCG64. Source: [Schrauwen et al. (2008), p. 1164](https://doi.org/10.1016/j.neucom.2007.12.020).

For unbounded recurrences, the benchmark accepts the first deterministically derived proposal whose target remains finite and `|y[t]| <= 1e6` over the complete split. Accepted inputs are therefore sampled from the stated proposal distribution **conditioned on this target-stability event**. Every rejected seed and acceptance horizon is recorded. Every stream is accepted over one common horizon equal to the maximum requested split length, and requested datasets are prefixes of those accepted streams.

The definitions differ across orders; their errors must not be interpreted as a pure order/memory scaling curve.

# Methane Plume Detection in Sentinel-2 SWIR Imagery

## Overview

The Sentinel-2 mission provides persistent multi-spectral imagery in the SWIR range and a two-to-ten-days revisit time. These bands are sensitive to the presence of atmospheric methane, thus enabling detection and quantification of large CH4 emissions. Varon et al. (2022) showed that methane concentration enhancements in a plume may be retrieved from multi-band satellites like Sentinel-2 using the multi-band multi-pass (MBMP) method. The MBMP algorithm compares observations of thesame scene with and without a methane plume using spectral bands that are sufficiently close to have similar surface and aerosol reflectance properties but differ in their methane absorption properties. Ehret et al. (2022) subsequently enhanced this approach by utilising regression analysis with time series data to compute background SWIR reflectance - as opposed to manually selecting a reference scene as background.

## Contents

- __notebooks/s2_demo.ipynb__ - detection of O&G super-emission events in Algeria, Turkmenistan and the Permian Basin, replicating test results reported by Varon et al. (2022)
- __notebooks/s2_demo_background.ipynb__ - rerun of Varon et. al. test case analysis using harmonic time series regression to compute background SWIR images.

## References

- Varon et. al. (2022), [High-frequency monitoring of anomalous methane point sources with multispectral Sentinel-2 satellite observations](https://doi.org/10.5194/amt-14-2771-2021)
- Pandey et. al. (2023), [Daily detection and quantification of methane leaks using Sentinel-3: a tiered satellite observation approach with Sentinel-2 and Sentinel-5p](https://doi.org/10.1016/j.rse.2023.113716)
- Ehret et. al. (2022), [Global Tracking and Quantification of Oil and Gas Methane Emissions from Recurrent Sentinel-2 Imagery](https://doi.org/10.48550/arXiv.2110.11832)

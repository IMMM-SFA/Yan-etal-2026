# yan-etal_2026

**Atmospheric and Land-Cover Shifts Enhance Terrestrial Productivity Resistance to Flash Droughts Across the United States**

Hongxiang Yan<sup>1\*</sup>, Lili Yao<sup>1</sup>, Ning Sun<sup>1</sup>, Eva Sinha<sup>1</sup>, Kanishka B. Narayan<sup>1</sup>, and Jennie Rice<sup>1</sup>

<sup>1</sup> Pacific Northwest National Laboratory, Richland, WA, USA

\* corresponding author: Hongxiang Yan, hongxiang.yan@pnnl.gov

## Abstract

Flash droughts are rapid-onset hydroclimatic extremes driven by compounding precipitation deficits and surging atmospheric vapor pressure deficits. While rising temperature accelerates soil moisture depletion, how terrestrial productivity resistance evolves across shifting atmospheric drivers and land cover remains uncertain. Using an integrated multi-sector modeling framework across the contiguous United States for the mid-century horizon (2021–2055), we show that under isolated atmospheric change, carbon enrichment enhances inherent water-use efficiency, providing a physiological buffer that counterintuitively increases terrestrial productivity resistance despite more frequent and severe flash droughts. Furthermore, dynamic land-use conversions actively amplify this resistance. Widespread conversion of grasslands to croplands shifts peak vegetative activity earlier in the season, facilitating a 'phenological escape' that decouples maximum carbon assimilation from late-summer hydroclimatic extremes. Ignoring these combined physiological and land-use buffers could lead models to overestimate future flash drought impacts. Our findings demonstrate that terrestrial productivity resistance is fundamentally modulated by human landscape configurations, providing a mechanistic basis for incorporating land-management pathways into Earth system projections.

## Journal reference

Yan, H., Yao, L., Sun, N., Sinha, E., Narayan, K. and Rice, J. (2026). Atmospheric and Land-Cover Shifts Enhance Terrestrial Productivity Resistance to Flash Droughts Across the United States. Submitted to *Nature Communications* – August 2026.

## Data Reference

### Input Data

| Dataset | URL | DOI |
| --- | --- | --- |
| TGW-WRF | https://tgw-data.msdlive.org/ | https://doi.org/10.1038/s41597-023-02485-5, https://doi.org/10.57931/1885756 |
| GCAM-SELECT-Demeter | https://data.msdlive.org/records/vy529-6eg15 | https://doi.org/10.57931/2502083 |

### Output Data

| Dataset | URL | DOI |
| --- | --- | --- |
| CLM5 soil moisture and GPP simulations | https://data.msdlive.org/uploads/v0j35-eqv54 | https://doi.org/10.57931/3420371 |

### Contributing Modeling Software

| Model | Version | URL | DOI |
| --- | --- | --- | --- |
| CLM5 | ctsm5.1.dev118 | https://github.com/IMMM-SFA/im3-clm | https://zenodo.org/records/6653705 |
| IM3 Components | 0cf45e8 | https://github.com/IMMM-SFA/im3components/tree/main/im3components/wrf_to_clm | |

## Reproduce my experiment

Clone the [CLM5 repository](https://github.com/ESCOMP/CTSM/tree/ctsm5.1.dev118) to set up the CLM5 model. You will need to download the [TGW forcing data](https://data.msdlive.org/records/ksw6r-2xv06) and convert them into CLM input format using these [scripts](https://github.com/IMMM-SFA/im3components/tree/main/im3components/wrf_to_clm). You will also need to replace the default CLM surface and land use timeseries files using data from the [GCAM-SELECT-Demeter](https://data.msdlive.org/records/vy529-6eg15). In addition, hydrological parameter values in the default parameter file and the user name list file should be updated based on the [behavioral parameter values](https://data.msdlive.org/records/41bw1-3q739). The [output data repository](https://data.msdlive.org/uploads/v0j35-eqv54) already contains the soil moisture and GPP output from the CLM5 model so you can skip rerunning the CLM5 model if you want to save time.

## Reproduce my figures

Use the scripts found in the `figures` directory to reproduce the figures used in this publication.

| Figure Numbers | Script Name | Description |
| --- | --- | --- |
| 1 | `Figure_1.py` | Spatial patterns of terrestrial productivity resistance to flash drought |
| 2 | `Figure_2.py` | Changes in growing season (April–October) flash drought characteristics |
| 3 | `Figure_3.py` | Changes in flash drought terrestrial productivity resistance and iWUE |
| 4 | `Figure_4.py` | Impact of LULC change on flash drought characteristics |
| 5 | `Figure_5.py` | Flash drought impacts on terrestrial productivity resistance |
| S1 | `Figure_S1.py` | Scatterplot of climatology/vegetation factors and flash drought ratio |
| S2 | `Figure_S2.py` | Scatterplot of land and soil factors and flash drought ratio |
| S3 | `Figure_S3.py` | Regional distribution of climatology factors across CONUS |
| S4 | `Figure_S4.py` | Regional distribution of atmospheric factors across CONUS |
| S5 | `Figure_S5.py` | Regional distribution of land cover factors across CONUS |
| S6 | `Figure_S6.py` | Regional distribution of vegetation factors across CONUS |
| S7 | `Figure_S7.py` | Regional distribution of soil factors across CONUS |
| S8 | `Figure_S8.py` | Feature importance from the EBM model |
| S9 | `Figure_S9.py` | Land cover among the historical and future scenarios |
| S10 | `Figure_S10.py` | GPP time series for cells undergoing grass-to-crop conversion |
| S11 | `Figure_S11.py` | Mean monthly VPD for grass-to-crop conversion cells |
| S12 | `Figure_S12.py` | CDFs of GPP decline rate for cells undergoing grass-to-forest conversion |
| S13 | `Figure_S13.py` | Illustration of flash versus slow-onset drought |
| S14 | `Figure_S14.py` | Illustrative flash drought event showing soil moisture and GPP percentile |

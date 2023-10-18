# soil-id-algorithm

## explanation of algorithm

### terminology

- soil map unit: (possibly disjoint) geographic area that is associated with soil component percentage / arial coverage
- soil series: collection of related soil components
- soil component: description of various soil properties at specific depth intervals

### references

- equation 1 in https://landpotential.org/wp-content/uploads/2020/07/sssaj-0-0-sssaj2017.09.0337.pdf

### dependencies

- simple features: https://r-spatial.github.io/sf/index.html
- well-known geometry: https://paleolimbot.github.io/wk/
- R package for querying soilDB: https://ncss-tech.github.io/soilDB/
- dplyr: https://dplyr.tidyverse.org/

### algorithm

Input: a specific point in lat/lon, and a set of depth intervals.

1. Query for all map units within 1km of the point.
2. Fall back to STATSGO at 10km if SSURGO is incomplete, or else declare not available if area not surveyed.
3. Associate each map unit with its polygons' minimum distance to the point in question.
4. Infill missing components by rescaling them to sum to 100.
5. Calculate the component probabilities by, for each component, dividing the distance-weighted sum of that component's probability in each map unit by the total distance-weighted sum of each component's probability in each map unit.
6. Limit to components in the top 12 component series by probability.
7. Query the local database for the component horizons.
8. Return the individual probabilities of data at each horizon based on the weighted sum of each component's data at each horizon.

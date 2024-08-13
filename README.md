# Soil ID Algorithm

## explanation of algorithm

### terminology

-   soil map unit: (possibly disjoint) geographic area that is associated with soil component percentage / arial coverage
-   soil series: collection of related soil components
-   soil component: description of various soil properties at specific depth intervals

### references

-   equation 1 in https://landpotential.org/wp-content/uploads/2020/07/sssaj-0-0-sssaj2017.09.0337.pdf

### dependencies

-   simple features: https://r-spatial.github.io/sf/index.html
-   well-known geometry: https://paleolimbot.github.io/wk/
-   R package for querying soilDB: https://ncss-tech.github.io/soilDB/
-   dplyr: https://dplyr.tidyverse.org/

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

### SoilID Project Box Folder

• This folder contains the data schema and processed soil database tables that are ingested into the mySQL database.

• https://nrcs.app.box.com/s/vs999nq9ruyetb9b4l7okmssdggh8okn

### SSURGO/STATSGO2 metadata:

• https://www.nrcs.usda.gov/resources/data-and-reports/ssurgo/stats2go-metadata

### SSURGO/STATSGO data:

• https://nrcs.app.box.com/v/soils/folder/17971946225

### Acknowledgements

* Beaudette, D., Roudier, P., Brown, A. (2023). [aqp: Algorithms for Quantitative Pedology](https://CRAN.R-project.org/package=aqp). R package version 2.0.
 
* Beaudette, D.E., Roudier, P., O'Geen, A.T. [Algorithms for quantitative pedology: A toolkit for soil scientists, Computers & Geosciences](http://dx.doi.org/10.1016/j.cageo.2012.10.020), Volume 52, March 2013, Pages 258-268, ISSN 0098-3004.
 
* soilDB: Beaudette, D., Skovlin, J., Roecker, S., Brown, A. (2024). [soilDB: Soil Database Interface](https://CRAN.R-project.org/package=soilDB). R package version 2.8.3.

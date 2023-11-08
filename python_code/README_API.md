# US API Endpoints Documentation

## Soilidlist Endpoint

This API provides detailed soil properties for a given latitude and longitude.

### URL
```
http://127.0.0.1:5000/api/v2/soilidlist
```

### Required Inputs
1. **longitude**: Longitude value.
   - Type: Float.
   - This input is mandatory, specifying the longitudinal coordinate for soil data retrieval.
2. **latitude**: Latitude value.
   - Type: Float.
   - This input is also mandatory and indicates the latitudinal coordinate for the desired soil data.

### Optional Inputs
3. **plot_id**: The LandPKS plot ID.
   - Type: Float.
   - This input is optional. If provided, it can help fetch specific soil data related to a particular plot or location.

### JSON Return
- **metadata**: Contains metadata about the soil data.
  - `location`: Location for which the data is applicable (e.g., "us").
  - `model`: Model version (e.g., "v2").
  - `unit_measure`: Units of measure for various parameters, like `cec`, `clay`, `depth`, `distance`, `ec`, `rock_fragments`, `sand`.
- **soilList**: A list detailing soil components.
  - Properties include `bottom_depth`, `cec`, `clay`, `ec`, `esd`, `id`, `lab`, `munsell`, `ph`, `rock_fragments`, `sand`, `texture`, `site`.

- **site**: Contains specific site data.
  - **siteData**: Detailed information about the soil component.
  - **siteDescription**: Textual description providing context and information about the soil component.

## Soilidrank Endpoint

The `soilidrank` API endpoint retrieves ranked soil data based on various properties.

### URL
```
http://127.0.0.1:5000/api/v2/soilidrank
```

### Required Inputs
1. **longitude**: Longitude value.
   - Type: Float.
2. **latitude**: Latitude value.
   - Type: Float.

### Optional Inputs
3. **plot_id**: The LandPKS plot ID.
   - Type: Float.
4. **soilHorizon[1-7]**: Soil texture for each of the seven possible horizons.
   - Options include textures like 'SILTY CLAY', 'SILT LOAM', etc.
5. **soilHorizon[1-7]_RFV**: Rock fragment volume for each horizon.
   - Classifications include '0-1%', '1-15%', '15-35%', etc.
6. **soilHorizon[1-7]_Depth**: Depth of each horizon in centimeters.
   - Type: Integer.
7. **soilHorizon[1-7]_LAB**: Dry color of each horizon.
   - Type: String.
8. **cracks**: Presence of deep, vertical cracks.
   - Type: Boolean.
9. **bedrock**: Depth to bedrock in centimeters.
   - Type: Integer.
10. **slope**: Slope percentage.
    - Type: Integer.
11. **elevation**: Elevation in meters.
    - Type: Integer.

### JSON Return
- **metadata**: Contains metadata about the data's completeness and suggestions for improvement.
  - `dataCompleteness`: Indicates the completeness of the data.
  - `location`: Location for which the data is applicable (e.g., "us").
  - `model`: Model version (e.g., "v3").
- **soilRank**: List containing rankings of soil components.
  - Details include `component`, `componentData`, `componentID`, `name`, `rank_data`, `rank_data_loc`, `rank_loc`, `score_data`, `score_data_loc`, `score_loc`.

This API provides rankings and scores for soil components based on data and location for specific latitude and longitude coordinates.

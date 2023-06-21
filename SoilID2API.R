# SoilID2API.R

source('C:/R_Drive/Data_Files/LPKS_Data/R_Projects/SoilID.2.0/code/base_functions_SoilID_US_API.R')

#* Run the SoilID location algorithm
#* @param lon Add longitude
#* @param lat Add latitude
#* @param intervals:[int] Add list
#* @get /soilID_list
# soilID list algorithm
function(lon, lat,intervals=c(0,10,20,50,70,100,120)){
  p = sf::st_sfc(sf::st_point(c(as.numeric(lon), as.numeric(lat))), crs = 4326)
  # transform to planar coordinate system for buffering
  p.aea <- sf::st_transform(p, crs= 5070)
  # create 1000 meter buffer
  p.aea_buff <- sf::st_buffer(p.aea, dist = 1000)
  # transform back to WGS84 GCS
  p.buff <- sf::st_transform(p.aea_buff, crs = 4326)
  wkt <- wk::wk_collection(wk::as_wkt(p.buff))

  # structure SDA query

  # query SSURGO first
  .template.SU.I <- "\n  WITH geom_data (geom, mukey) AS (\n  SELECT\n  mupolygongeo.STIntersection( geometry::STGeomFromText('%s', 4326) ) AS geom, P.mukey\n  FROM mupolygon AS P\n  WHERE mupolygongeo.STIntersects( geometry::STGeomFromText('%s', 4326) ) = 1\n)\nSELECT\ngeom.STAsText() AS geom, mukey,\nGEOGRAPHY::STGeomFromWKB(\n    geom.STUnion(geom.STStartPoint()).STAsBinary(), 4326).STArea() * 0.000247105 AS area_ac\nFROM geom_data;\n  "
  q <- sprintf(.template.SU.I, as.character(wkt), as.character(wkt))
  mu_data <- suppressMessages(soilDB::SDA_query(q))

  # check if area is 'NOTCOM' or 'AREA NOT SURVEYED'
  if(nrow(mu_data)==1){
    area_check <- suppressMessages(soilDB::SDA_query(sprintf("SELECT mukey, compname FROM component WHERE mukey = %s", mu_data$mukey))$compname)
    if(area_check=="NOTCOM"){
      data_source = 'STATSGO'
      # create 10000 meter buffer
      p.aea_buff <- sf::st_buffer(p.aea, dist = 10000)
      # transform back to WGS84 GCS
      p.buff <- sf::st_transform(p.aea_buff, crs = 4326)
      wkt <- wk::wk_collection(wk::as_wkt(p.buff))
      .template.ST.I <- "\n  WITH geom_data (geom, mukey) AS (\n  SELECT\n  mupolygongeo.STIntersection( geometry::STGeomFromText('%s', 4326) ) AS geom, P.mukey\n  FROM gsmmupolygon AS P\n  WHERE mupolygongeo.STIntersects( geometry::STGeomFromText('%s', 4326) ) = 1\n  AND CLIPAREASYMBOL = 'US'\n)\nSELECT\ngeom.STAsText() AS geom, mukey,\nGEOGRAPHY::STGeomFromWKB(\n    geom.STUnion(geom.STStartPoint()).STAsBinary(), 4326).STArea() * 0.000247105 AS area_ac\nFROM geom_data;\n  "
      q <- sprintf(.template.ST.I, as.character(wkt), as.character(wkt))
      mu_data <- suppressMessages(soilDB::SDA_query(q))
      mu_data <- soilDB::processSDA_WKT(mu_data, as_sf = TRUE)
      mu_data <- mu_data |> dplyr::mutate(distance = as.numeric(sf::st_distance(p, mu_data)))|> sf::st_set_geometry(NULL)
    } else if(area_check=="AREA NOT SURVEYED"){
      return("Soil ID not available in this area")
    } else {
      data_source = 'SSURGO'
      mu_data <- soilDB::processSDA_WKT(mu_data, as_sf = TRUE)
      mu_data <- mu_data |> dplyr::mutate(distance = as.numeric(sf::st_distance(p, mu_data)))|> sf::st_set_geometry(NULL)
    }
  } else {
    data_source = 'SSURGO'
    mu_data <- soilDB::processSDA_WKT(mu_data, as_sf = TRUE)
    mu_data <- mu_data |> dplyr::mutate(distance = as.numeric(sf::st_distance(p, mu_data)))|> sf::st_set_geometry(NULL)
  }

  #Some mapunits may have multiple polygon geometries. Need to find the shortest distance to closest polygon
  mu_data <- mu_data |> dplyr::group_by(mukey) |> dplyr::mutate(area_ac = sum(area_ac), distance = min(distance)) |> dplyr::ungroup() |> dplyr::distinct() |> as.data.frame()
  mucompdataQry =  sprintf('SELECT component.mukey, component.cokey,  component.compname,  component.comppct_r, component.compkind, component.majcompflag,  component.slope_r,  component.slope_l,  component.slope_h, component.elev_r, component.elev_l, component.elev_h, component.nirrcapcl, component.nirrcapscl, component.nirrcapunit, component.irrcapcl, component.irrcapscl, component.irrcapunit, component.taxorder, component.taxsubgrp FROM component WHERE mukey IN ( %s )', paste(mu_data$mukey, sep="'", collapse=", "))
  comp_data <- suppressMessages(soilDB::SDA_query(mucompdataQry))
  comp_data <- comp_data |> dplyr::left_join(mu_data, by="mukey")
  if(data_source == 'STATSGO'){
    ExpCoeff = -0.0002772 #Expotential decay coefficient: 0.25 @ ~5km
  } else if(data_source == 'SSURGO'){
    ExpCoeff = -0.008 #Old expotential decay coefficient: 0.25 @ ~175m
    #ExpCoeff = -0.002772 #New expotential decay coefficient. Decreases to .25 at 500 meters
  }

  #comppct_r check -- infill missing values and ensure all comps within mapunit sum to 100
  comp_data <- comp_data |> dplyr::group_by(mukey) |> dplyr::mutate(comppct_r = comppct_r/sum(comppct_r)*100) |> dplyr::ungroup() |> dplyr::distinct() |> as.data.frame()

  #--------------------------------------------------------------------------------------------------
  #Location based calculation
  #-----------------------------

  #--------------------------------------------------------------------------------------------------------------------------------------------------------
  #########################################################################################################################################
  # Individual probability
  #Based on Fan et al 2018 EQ 1, the conditional probability for each component is calculated by taking the sum of all occurances of a component
  #in the home and adjacent mapunits and dividing this by the sum of all map units and components. We have modified this approach so that each
  #instance of a component occurance is evaluated separately and assinged a weight and the max distance score for each component group is assigned
  #to all component instances.
  #########################################################################################################################################

  comp_data <- comp_data |> dplyr::group_by(mukey,cokey) |>
    dplyr::mutate(loc_score = dplyr::case_when(
      distance == 0 ~ sum(comppct_r)/100
      ,distance > 0 & exp(ExpCoeff*distance) < 0.25 ~ sum(comppct_r)/100 * 0.25
      ,TRUE ~  sum(comppct_r)/100 * exp(ExpCoeff*distance))) |> dplyr::ungroup() |> dplyr::distinct() |> as.data.frame()

  comp_data <- comp_data |> dplyr::mutate(loc_score_sum = sum(loc_score)) |> dplyr::group_by(compname) |> dplyr::mutate(loc_score_comp = sum(loc_score)/loc_score_sum) |> dplyr::ungroup() |> dplyr::mutate(loc_score = loc_score/loc_score_sum) |> dplyr::mutate(loc_score_comp = loc_score_comp/max(loc_score_comp)) |> dplyr::distinct() |> as.data.frame()
  comp_data <- comp_data |> dplyr::group_by(compname) |> dplyr::mutate(loc_score_max = max(loc_score)) |> dplyr::ungroup() |> dplyr::distinct() |> as.data.frame() |> dplyr::select(-loc_score_sum)

  #Limit number of components to top 12
  if(length(unique(comp_data$compname)) > 12) {
    top_comp <- comp_data |> dplyr::select(compname, loc_score_comp) |> dplyr::distinct() |> dplyr::slice_max(order_by = loc_score_comp, n = 12) |> dplyr::select(compname) |> dplyr::pull()
    comp_data <- comp_data |> dplyr::filter(compname %in% top_comp)
  }
  if(data_source == 'SSURGO'){
    comp_data <- comp_data |> dplyr::mutate(data_source = 'SSURGO')
  } else if (data_source == 'STATSGO'){
    comp_data <- comp_data |> dplyr::mutate(data_source = 'STATSGO')
  }

  #-------------------------------------------------------------------------------------------------------------------
  # Query component horizon data from local database

  # open DB connection, run query, close connection
  pool <- pool::dbPool(
    drv = RMySQL::MySQL(),
    dbname = "apex",
    host = "127.0.0.1",
    username = "root" ,
    password = "root"
  )
  q <- "SELECT mukey, cokey, top, bottom, claytotal_r, sandtotal_r, silttotal_r, fragvol_r, ph1to1h2o_r, om_r, awc_r, dbovendry_r, claytotal_l, sandtotal_l, silttotal_l, fragvol_l, ph1to1h2o_l, om_l, awc_l, dbovendry_l,claytotal_h, sandtotal_h, silttotal_h, fragvol_h, ph1to1h2o_h, om_h, awc_h, dbovendry_h, compname, restLyr, comppct_r FROM ssurgo_cokey_ranges_LPKS WHERE cokey IN (%s)"
  query <- sprintf(q, paste(comp_data$cokey,collapse = ","))
  hz_data = DBI::dbGetQuery(pool, query)
  pool::poolClose(pool)

  # count number of component-depth replicates
  hz_data <- hz_data |> dplyr::group_by(compname,bottom) |> dplyr::mutate(rep_n = length(compname) |> as.numeric()) |> dplyr::ungroup() |> as.data.frame()

  hz_stats <- hz_quant_prob(hz_data)

  hz_stats$compname = ordered(hz_stats$compname, levels=c(unique(hz_data$compname)))
  hz_json <- jsonlite::toJSON(hz_stats |> dplyr::group_split(compname))
  hz_json2 <- jsonlite::fromJSON(hz_json)
  comp_data$compname = ordered(comp_data$compname, levels=c(unique(hz_data$compname)))
  comp_data <- comp_data |> dplyr::arrange(factor(compname))
  comp_json <- jsonlite::toJSON(comp_data |> dplyr::group_split(compname))
  comp_json2 <- jsonlite::fromJSON(comp_json)
  data_out <- list(list())
  for(i in 1:length(comp_json2)){
    x <- c(comp_json2[[i]], hz_json2[[i]])
    x <- setNames(list(x), comp_json2[[i]]$compname[1])
    data_out[[i]] <- x
  }

  return(data_out)
}

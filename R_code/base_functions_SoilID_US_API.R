## Base functions for LandPKS-US SoilID 2.0 API processing
## Author: Jonathan Maynard
## Date: July 15, 2022


# calculate texture class from sand, silt, clay percentages. Example input: (20,30,50) = 20% sand, 30% silt, 50% clay.
gettt <- function(sand, silt, clay){
  if(is.na(sand) | is.na(silt) | is.na(clay)){
    x = NA
  } else if((silt + 1.5 * clay) < 15){
    x = "Sand"
  } else if((silt + 1.5 * clay) >= 15 & (silt + 2.0 * clay) < 30){
    x = "Loamy sand"
  } else if((clay >= 7) & (clay <= 20) & (sand > 52) & ((silt + 2.0 * clay) >= 30)){
    x = "Sandy loam"
  } else if((clay < 7) & (silt < 50) & ((silt + 2.0 * clay) >= 30)){
    x = "Sandy loam"
  } else if((clay >= 7) & (clay <= 27) & (silt >= 28) & (silt < 50) & (sand <= 52)){
    x = "Loam"
  } else if(((silt >= 50) & (clay >= 12) & (clay < 27)) | ((silt >= 50) & (silt < 80) & (clay < 12))){
    x = "Silt loam"
  } else if((silt >= 80) & (clay < 12)){
    x = "Silt"
  } else if((clay >= 20) & (clay < 35) & (silt < 28) & (sand > 45)){
    x = "Sandy clay loam"
  } else if((clay >= 27) & (clay < 40) & (sand > 20) & (sand <= 45)){
    x = "Clay loam"
  } else if((clay >= 27) & (clay < 40) & (sand <= 20)){
    x = "Silty clay loam"
  } else if((clay >= 35) & (sand >= 45)){
    x = "Sandy clay"
  } else if((clay >= 40) & (silt >= 40)){
    x = "Silty clay"
  } else if((clay >= 40) & (sand <= 45) & (silt < 40)){
    x = "Clay"
  }
  return(x)
}

# calculate rock fragment group. Example input: (40) = 40% rock fragments
getCF_groups <- function(cf){
  if(is.na(cf)){
    cf_g <-  NA
  }else if(cf >= 0 & cf < 2){
    cf_g <-  "0-1%"
  }else if( cf >=2 & cf < 16){
    cf_g <-  "1-15%"
  }else if(cf >= 16 & cf < 36){
    cf_g <-  "15-35%"
  }else if(cf >= 36 & cf < 61){
    cf_g <-  "35-60%"
  }else if(cf >= 61){
    cf_g <-  ">60%"
  }
  return(cf_g)
}

# Remove any pedon horizons below bedrock depth
hrz_bdrock_clip <- function(pedon){
  if(any(!is.na(pedon$bedrkdepth))){
    if(max(pedon$hzdepb_r) > unique(pedon$bedrkdepth)){
      remove_row <- which(c(pedon$hzdepb_r) >= unique(pedon$bedrkdepth))
      if(length(remove_row)==1){
        pedon$hzdepb_r[remove_row] <- unique(pedon$bedrkdepth)
      } else {
        pedon$hzdepb_r[remove_row[1]] <- unique(pedon$bedrkdepth)
        pedon <- pedon[-c(remove_row[2:length(remove_row)]),]
      }
    }
  }
  return(pedon)
}

# list of bedrock horizon names
bedrock_hzname = c("3R", "2R", "R", "3Rt", "R3", "R2", "R4", "4R", "Rk", "CR", "5R", "Rkq", "R, Cd", "2Rj", "R5", "2Bt3/2CR", "9R", "2Cr,2R", "3Cr,3R", "C/R", "Rr", "Cr,R", "2C,3R", "R1", "6R", "A, R", "IICR", "Rt", "2R4",  "R6", "3Rb", "2R3", "2Cr,R", "4Cr,4R", "R/Cr", "Cr,2R", "3Cr", "2Cr", "Cr", "Crt", "2Crk", "2Crt", "2Cr1", "2Cr2", "3Crk", "2Crf", "2Crtk", "4Cr", "Cr1,Cr2", "Crk", "Crkq", "2Crkq", "Crtk", "Cr2", "Cr1", "Cr3", "2Cr1,2Cr2", "2Cr,2R", "3Cr1,3Cr2", "2Cr1-2Cr3", "Cr4", "Cr/B", "3Crkq", "5Cr", "2Cr,R", "4Cr,4R", "R/Cr", "Cr,2R","Cd","Cd1", "Cd2")

mukey_sim <- function(mukey_data, props = "LPKS"){
  data <- list(list())
  for(i in 1:nrow(mukey_data)){
    data_in <- mukey_data[i,]
    sim_n <- round(100/data_in$rep_n, 0)
    data[[i]] <- prop_sim(mukey_data[i,], reps=sim_n, properties= props)

  }
  data_out <- dplyr::bind_rows(data)
  return(data_out)
}

mukey_sim_comp <- function(mukey_data, props = "LPKS"){
  data <- list(list())
  for(i in 1:nrow(mukey_data)){
    data_in <- mukey_data[i,]
    sim_n <- round(100/data_in$rep_n, 0)
    data[[i]] <- prop_sim(mukey_data[i,], reps=sim_n, properties=props)
    #data_out <- dplyr::bind_rows(data)
  }
  return(data)
}

hz_quant_prob <- function(hz_data, top_prob_num=1){
  data_bind <- mukey_sim(hz_data, props = "ALL")
  q = c(0.05, 0.5, 0.95)
  data_out <- data_bind |> dplyr::select(compname, top, sand=sandtotal_r, silt=silttotal_r, clay=claytotal_r, rfv=fragvol_r, ph=ph1to1h2o_r, om=om_r, awc=awc_r, db=dbovendry_r)

  data_stats05 <- data_out |> dplyr::group_by(compname, top) |> dplyr::summarize(across(sand:db, ~quantile(.x, probs = q[1]))) |> as.data.frame()
  data_stats50 <- data_out |> dplyr::group_by(compname, top) |> dplyr::summarize(across(sand:db, ~quantile(.x, probs = q[2]))) |> as.data.frame()
  data_stats95 <- data_out |> dplyr::group_by(compname, top) |> dplyr::summarize(across(sand:db, ~quantile(.x, probs = q[3]))) |> as.data.frame()
  data_statsPIW90 <- data_stats95 |> dplyr::select(-c(compname)) - data_stats05 |> dplyr::select(-c(compname))
  data_statsPIW90 <- data_statsPIW90  |> dplyr::mutate(compname=data_stats05$compname, .before = top) |> dplyr::mutate(top=data_stats05$top)
  data_stats05 <- data_stats05 |> purrr::set_names(c("compname", "top", "sand_05", "silt_05", "clay_05", "rfv_05",  "ph_05", "om_05", "awc_05", "db_05"))
  data_stats50 <- data_stats50 |> purrr::set_names(c("compname", "top", "sand_50", "silt_50", "clay_50", "rfv_50",  "ph_50", "om_50", "awc_50", "db_50"))
  data_stats95 <- data_stats95 |> purrr::set_names(c("compname", "top", "sand_95", "silt_95", "clay_95", "rfv_95",  "ph_95", "om_95", "awc_95", "db_95"))
  data_statsPIW90 <- data_statsPIW90 |> purrr::set_names(c("compname", "top", "sand_PIW90", "silt_PIW90", "clay_PIW90", "rfv_PIW90",  "ph_PIW90", "om_PIW90", "awc_PIW90", "db_PIW90"))
  data_stats <- data_stats05 |> dplyr::left_join(data_stats50, by=c("compname"="compname", "top"="top"))
  data_stats <- data_stats |> dplyr::left_join(data_stats95, by=c("compname"="compname", "top"="top"))
  data_stats <- data_stats |> dplyr::left_join(data_statsPIW90, by=c("compname"="compname", "top"="top"))
  data_stats <- data_stats |> dplyr::select(compname, top, colnames(data_stats)[-1:-2][order(colnames(data_stats)[-1:-2])])
  data_stats <- hz_data |> dplyr::select(compname, top, bottom) |> dplyr::distinct() |> dplyr::left_join(data_stats, by=c("compname"="compname", "top"="top"))

  #txt probabilities
  txt_out <- data_out |> dplyr::select(compname, top, SAND = sand, SILT = silt, CLAY = clay)
  txt_out$txt_class <- soiltexture::TT.points.in.classes(tri.data=txt_out, class.sys="USDA-NCSS.TT", PiC.type="t")
  txt_class_prob <- txt_out |> dplyr::group_by(compname) |> dplyr::count(top, txt_class) |> dplyr::mutate(prob=n/100) |> dplyr::ungroup() |> dplyr::group_by(compname,top)  |> dplyr::slice_max(prob, n=top_prob_num) |> dplyr::ungroup() |> as.data.frame()
  txt_class_prob <- hz_data |> dplyr::select(compname, top, bottom) |> dplyr::distinct() |> dplyr::left_join(txt_class_prob, by=c("compname"="compname", "top"="top"))

  data_stats <- data_stats |> dplyr::left_join(txt_class_prob |> dplyr::select(-c(n)), by=c("compname"="compname", "top"="top", "bottom"="bottom"))

  return(data_stats)
}


mukey_txt_prob_top <- function(mukey_data, top_prob_num){
  data <- list(list())
  for(i in 1:nrow(mukey_data)){
    data[[i]] <- prop_sim_loc_wt(mukey_data[i,])
  }
  data_out <- dplyr::bind_rows(data)
  data_out <- data_out |> dplyr::select(top, SAND = sandtotal_r, SILT = silttotal_r, CLAY = claytotal_r)
  data_out$txt_class <- soiltexture::TT.points.in.classes(tri.data=data_out, class.sys="USDA-NCSS.TT", PiC.type="t")
  txt_class_prob <- data_out |> dplyr::count(top, txt_class) |> dplyr::mutate(prob=n/50) |> dplyr::group_by(top)  |> dplyr::slice_max(prob, n=top_prob_num) |> dplyr::ungroup() |> as.data.frame()
  txt_class_prob <- mukey_data |> dplyr::select(mukey, cokey, compname, top, bottom) |> dplyr::distinct() |> dplyr::left_join(txt_class_prob, by="top")
  return(txt_class_prob)
}

mukey_txt_prob_all <- function(mukey_data){
  data <- list(list())
  for(i in 1:nrow(mukey_data)){
    data[[i]] <- prop_sim_loc_wt(mukey_data[i,])
  }
  data_out <- dplyr::bind_rows(data)
  data_out <- data_out |> dplyr::select(SAND = sandtotal_r, SILT = silttotal_r, CLAY = claytotal_r)
  txt_class <- soiltexture::TT.points.in.classes(tri.data=data_out, class.sys="USDA.TT")
  txt_class_m = txt_class |> colMeans() |> as.data.frame()
  txt_class_m <- txt_class_m |> dplyr::mutate(txt_cl = rownames(txt_class_m), txt_num = seq(1, nrow(txt_class_m)), mukey=mukey_data |> dplyr::select(mukey) |> dplyr::distinct() |> dplyr::pull()) |> purrr::set_names("prob" , 'txt_cl', 'txt_num', 'mukey')

  return(txt_class_m)
}

# Simulate data based on data range weighted by location score
prop_sim_loc_wt <- function(data){
  ssc <- compositions::acomp(data[, c('sandtotal_r', 'silttotal_r', 'claytotal_r')])
  # make fake samples using the overy-simplistic assumption that
  # the low -- high ranges represent a uniform distribution
  # ideally, low / rv / high would be referenced to a distributional benchmark (e.g. quantiles)
  # so that simulation would be more realistic, oh well.
  # result is a composition
  ssc.sim.range <- compositions::acomp(
    cbind(
      sand_range=runif(1000, min=data$sandtotal_l, max=data$sandtotal_h),
      silt_range=runif(1000, min=data$silttotal_l, max=data$silttotal_h),
      clay_range=runif(1000, min=data$claytotal_l, max=data$claytotal_h)
    )
  )
  # nothing done here, but would safely compute the mean if nrow > 1
  texture.mean <- compositions::meanCol(ssc)

  # safely compute the covariance matrix from a composition
  texture.var <- var(ssc.sim.range)

  # draw 100 samples from an ideal composition with normal distributions
  # parameterized from our estimates above
  ssc.sim <- compositions::rnorm.acomp(n=round(data$loc_score_comp*100*50), mean=texture.mean, var=texture.var)

  # convert back to {0,100} scale for plotting
  ssc.sim <- as.data.frame(unclass(ssc.sim) * 100)
  ssc.sim$id <- data$id
  ssc.sim$compname <- data$compname
  ssc.sim$cokey <- data$cokey
  ssc.sim$mukey <- data$mukey
  #ssc.sim$Depth  <- data$Depth
  ssc.sim$top  <- data$top
  ssc.sim$bottom  <- data$bottom

  #calculate rfv simulated range
  rfv_range=runif(1000, min=data$fragvol_l, max=data$fragvol_h)
  rfv_mean = mean(rfv_range)
  rfv_sd = sd(rfv_range)
  rfv.sim <- rnorm(n=round(data$loc_score_comp*100*50), mean=rfv_mean, sd=rfv_sd)
  rfv.sim[rfv.sim < 0] <-  0.1
  ssc.sim$fragvol_r <- rfv.sim

  #calculate ph1to1h2o_r simulated range
  ph_range=runif(1000, min=data$ph1to1h2o_l, max=data$ph1to1h2o_h)
  ph_mean = mean(ph_range)
  ph_sd = sd(ph_range)
  ph.sim <- rnorm(n=round(data$loc_score_comp*100*50), mean=ph_mean, sd=ph_sd)
  ph.sim[ph.sim < 0] <-  0.1
  ssc.sim$ph1to1h2o_r <- ph.sim

  #calculate om simulated range
  om_range=runif(1000, min=data$om_l, max=data$om_h)
  om_mean = mean(om_range)
  om_sd = sd(om_range)
  om.sim <- rnorm(n=round(data$loc_score_comp*100*50), mean=om_mean, sd=om_sd)
  om.sim[om.sim < 0] <-  0.1
  ssc.sim$om_r <- om.sim

  #calculate awc simulated range
  awc_range=runif(1000, min=data$awc_l, max=data$awc_h)
  awc_mean = mean(awc_range)
  awc_sd = sd(awc_range)
  awc.sim <- rnorm(n=round(data$loc_score_comp*100*50), mean=awc_mean, sd=awc_sd)
  awc.sim[awc.sim < 0] <-  0.1
  ssc.sim$awc_r <- awc.sim

  #calculate db simulated range
  db_range=runif(1000, min=data$dbovendry_l, max=data$dbovendry_h)
  db_mean = mean(db_range)
  db_sd = sd(db_range)
  db.sim <- rnorm(n=round(data$loc_score_comp*100*50), mean=db_mean, sd=db_sd)
  db.sim[db.sim < 0] <-  0.1
  ssc.sim$dbovendry_r <- db.sim

  return(ssc.sim)
}

# calculate mean with NA values present
mean_na <- function(x){
  x_mean <- mean(x, na.rm=TRUE)
  return(x_mean)
}


prop_sim <- function(data, reps, properties){
  ssc <- compositions::acomp(data[, c('sandtotal_r', 'silttotal_r', 'claytotal_r')])
  # make fake samples using the overy-simplistic assumption that
  # the low -- high ranges represent a uniform distribution
  # ideally, low / rv / high would be referenced to a distributional benchmark (e.g. quantiles)
  # so that simulation would be more realistic, oh well.
  # result is a composition
  ssc.sim.range <- compositions::acomp(
    cbind(
      sand_range=runif(1000, min=data$sandtotal_l, max=data$sandtotal_h),
      silt_range=runif(1000, min=data$silttotal_l, max=data$silttotal_h),
      clay_range=runif(1000, min=data$claytotal_l, max=data$claytotal_h)
    )
  )
  # nothing done here, but would safely compute the mean if nrow > 1
  texture.mean <- compositions::meanCol(ssc)

  # safely compute the covariance matrix from a composition
  texture.var <- var(ssc.sim.range)

  # draw 100 samples from an ideal composition with normal distributions
  # parameterized from our estimates above
  ssc.sim <- compositions::rnorm.acomp(n=reps, mean=texture.mean, var=texture.var)

  # convert back to {0,100} scale for plotting
  ssc.sim <- as.data.frame(unclass(ssc.sim) * 100)
  ssc.sim$id <- data$id
  ssc.sim$compname <- data$compname
  ssc.sim$cokey <- data$cokey
  ssc.sim$mukey <- data$mukey
  #ssc.sim$Depth  <- data$Depth
  ssc.sim$top  <- data$top
  ssc.sim$bottom  <- data$bottom

  #calculate rfv simulated range
  rfv_range=runif(1000, min=data$fragvol_l, max=data$fragvol_h)
  rfv_mean = mean(rfv_range)
  rfv_sd = sd(rfv_range)
  rfv.sim <- rnorm(n=reps, mean=rfv_mean, sd=rfv_sd)
  rfv.sim[rfv.sim < 0] <-  0.1
  ssc.sim$fragvol_r <- rfv.sim

  if(properties=="ALL"){
    #calculate ph1to1h2o_r simulated range
    ph_range=runif(1000, min=data$ph1to1h2o_l, max=data$ph1to1h2o_h)
    ph_mean = mean(ph_range)
    ph_sd = sd(ph_range)
    ph.sim <- rnorm(n=reps, mean=ph_mean, sd=ph_sd)
    ph.sim[ph.sim < 0] <-  0.1
    ssc.sim$ph1to1h2o_r <- ph.sim

    #calculate om simulated range
    om_range=runif(1000, min=data$om_l, max=data$om_h)
    om_mean = mean(om_range)
    om_sd = sd(om_range)
    om.sim <- rnorm(n=reps, mean=om_mean, sd=om_sd)
    om.sim[om.sim < 0] <-  0.1
    ssc.sim$om_r <- om.sim

    #calculate awc simulated range
    awc_range=runif(1000, min=data$awc_l, max=data$awc_h)
    awc_mean = mean(awc_range)
    awc_sd = sd(awc_range)
    awc.sim <- rnorm(n=reps, mean=awc_mean, sd=awc_sd)
    awc.sim[awc.sim < 0] <-  0.1
    ssc.sim$awc_r <- awc.sim

    #calculate db simulated range
    db_range=runif(1000, min=data$dbovendry_l, max=data$dbovendry_h)
    db_mean = mean(db_range)
    db_sd = sd(db_range)
    db.sim <- rnorm(n=reps, mean=db_mean, sd=db_sd)
    db.sim[db.sim < 0] <-  0.1
    ssc.sim$dbovendry_r <- db.sim
  }
  return(ssc.sim)
}

# calculate mean with NA values present
mean_na <- function(x){
  x_mean <- mean(x, na.rm=TRUE)
  return(x_mean)
}









# create standardized depth intervals by specified depths. specify properties to include
# input parameters
# intervals <- c(0,20,50,80,120)
# properties = "ALL"
# data = hz_data

std_depth_sim <- function(data, intervals = c(0,20,50,80,120), properties = "ALL"){
  #subset properties for modeling and standardize depth intervals
  if(properties == "ALL"){
    data_sub <- data |> dplyr::select(cokey, top, bottom, sandtotal_l, sandtotal_h, sandtotal_r, silttotal_l, silttotal_h, silttotal_r, claytotal_l, claytotal_h, claytotal_r,
                                      fragvol_l, fragvol_h, fragvol_r, ph1to1h2o_l, ph1to1h2o_h, ph1to1h2o_r, om_l, om_h, om_r, awc_l, awc_h, awc_r, dbovendry_l, dbovendry_h, dbovendry_r)
  } else if(properties == "LPKS"){
    data_sub <- data |> dplyr::select(cokey, top, bottom, sandtotal_l, sandtotal_h, sandtotal_r, silttotal_l, silttotal_h, silttotal_r, claytotal_l, claytotal_h, claytotal_r,
                                      fragvol_l, fragvol_h, fragvol_r)
  }
  aqp::depths(data_sub) <-  cokey ~ top + bottom

  if(properties == "ALL"){
    data_profile_slab <- aqp::slab(data_sub, fm = cokey ~ sandtotal_l + sandtotal_h + sandtotal_r + silttotal_l + silttotal_h + silttotal_r + claytotal_l + claytotal_h + claytotal_r + fragvol_l + fragvol_h + fragvol_r + ph1to1h2o_l + ph1to1h2o_h + ph1to1h2o_r + om_l + om_h + om_r + awc_l + awc_h + awc_r + dbovendry_l + dbovendry_h + dbovendry_r, slab.structure=intervals, slab.fun=mean_na)
  } else if(properties == "LPKS"){
    data_profile_slab <- aqp::slab(data_sub, fm = cokey ~ sandtotal_l + sandtotal_h + sandtotal_r + silttotal_l + silttotal_h + silttotal_r + claytotal_l + claytotal_h + claytotal_r + fragvol_l + fragvol_h + fragvol_r, slab.structure=intervals, slab.fun=mean_na)
  }

  data_profile_slab <- data_profile_slab |> dplyr::select(-c(contributing_fraction)) |> tidyr::pivot_wider(names_from = variable, values_from = value)
  data_profile_slab[ is.na(data_profile_slab) ] <- NA
  max_depth <- data_profile_slab |> dplyr::group_by(cokey) |> dplyr::filter(!is.na(sandtotal_r) | !is.na(silttotal_r)| !is.na(claytotal_r)| !is.na(fragvol_r)) |> dplyr::select(bottom) |> dplyr::filter(bottom == max(bottom))

  data_profile_list <- list(list())
  for(i in 1:nrow(max_depth)){
    data_profile_list[[i]] <- data_profile_slab |> dplyr::filter(cokey==max_depth$cokey[i]) |> dplyr::filter(bottom <= max_depth$bottom[i])
  }
  data_profile_std <-  dplyr::bind_rows(data_profile_list) |> as.data.frame()
  data_profile_std <- data_profile_std |> dplyr::mutate(silttotal_l = replace(silttotal_l, silttotal_l<0, 0), silttotal_h = replace(silttotal_h, silttotal_h<0, 0), sandtotal_l = replace(sandtotal_l, sandtotal_l<0, 0), sandtotal_h = replace(sandtotal_h, sandtotal_h<0, 0))
  data_profile_std[is.na(data_profile_std)] <- 0.01
  data_profile_std[data_profile_std == -9999.00000] <-  0.01
  data_profile_std[,3:15][data_profile_std[,3:15] == 0] <-  0.01

  data_profile_std$cokey <- as.integer(data_profile_std$cokey)
  data_profile_std <- data_profile_std |> dplyr::left_join(data |> dplyr::select(cokey, mukey, compname, restLyr, comppct_r, rep_n) |> dplyr::distinct(), by="cokey")

  # simulate profile data
  sim_soil <- mukey_sim(data_profile_std)

  top <- intervals[1:(length(intervals)-1)]
  bottom <- intervals[2:length(intervals)]
  Depth <- list()
  for(i in 1:length(top)){
    Depth[i] <- paste0(top[i], "-", bottom[i], " cm")
  }
  depth_intervals <- data.frame(unlist(Depth), top) |> purrr::set_names("Depth", 'top')

  sim_soil <-  sim_soil  |> dplyr::left_join(depth_intervals, by="top")
  sim_soil <-  sim_soil |> dplyr::group_by(compname, Depth) |> dplyr::mutate(id = dplyr::row_number()) |> dplyr::ungroup()
  if(properties == "ALL"){
    sim_data_wide <- sim_soil |> dplyr::select(-c(top,bottom)) |> tidyr::pivot_wider(names_from = Depth, values_from = c(sandtotal_r, silttotal_r,claytotal_r,fragvol_r,ph1to1h2o_r,om_r,awc_r,dbovendry_r))
  } else if(properties == "LPKS"){
    sim_data_wide <- sim_soil |> dplyr::select(-c(top,bottom)) |> tidyr::pivot_wider(names_from = Depth, values_from = c(sandtotal_r, silttotal_r,claytotal_r,fragvol_r))
  }
  sim_data_wide[is.na(sim_data_wide)] <- 0.01
return(sim_data_wide)
}

std_depth_intervals <- function(data, intervals = c(0,20,50,80,120), properties = "ALL"){
  #subset properties for modeling and standardize depth intervals
  if(properties == "ALL"){
    data_sub <- data |> dplyr::select(cokey, top, bottom, sandtotal_l, sandtotal_h, sandtotal_r, silttotal_l, silttotal_h, silttotal_r, claytotal_l, claytotal_h, claytotal_r,
                                      fragvol_l, fragvol_h, fragvol_r, ph1to1h2o_l, ph1to1h2o_h, ph1to1h2o_r, om_l, om_h, om_r, awc_l, awc_h, awc_r, dbovendry_l, dbovendry_h, dbovendry_r)
  } else if(properties == "LPKS"){
    data_sub <- data |> dplyr::select(cokey, top, bottom, sandtotal_l, sandtotal_h, sandtotal_r, silttotal_l, silttotal_h, silttotal_r, claytotal_l, claytotal_h, claytotal_r,
                                      fragvol_l, fragvol_h, fragvol_r)
  }
  aqp::depths(data_sub) <-  cokey ~ top + bottom

  if(properties == "ALL"){
    data_profile_slab <- aqp::slab(data_sub, fm = cokey ~ sandtotal_l + sandtotal_h + sandtotal_r + silttotal_l + silttotal_h + silttotal_r + claytotal_l + claytotal_h + claytotal_r + fragvol_l + fragvol_h + fragvol_r + ph1to1h2o_l + ph1to1h2o_h + ph1to1h2o_r + om_l + om_h + om_r + awc_l + awc_h + awc_r + dbovendry_l + dbovendry_h + dbovendry_r, slab.structure=intervals, slab.fun=mean_na)
  } else if(properties == "LPKS"){
    data_profile_slab <- aqp::slab(data_sub, fm = cokey ~ sandtotal_l + sandtotal_h + sandtotal_r + silttotal_l + silttotal_h + silttotal_r + claytotal_l + claytotal_h + claytotal_r + fragvol_l + fragvol_h + fragvol_r, slab.structure=intervals, slab.fun=mean_na)
  }

  data_profile_slab <- data_profile_slab |> dplyr::select(-c(contributing_fraction)) |> tidyr::pivot_wider(names_from = variable, values_from = value)
  data_profile_slab[ is.na(data_profile_slab) ] <- NA
  max_depth <- data_profile_slab |> dplyr::group_by(cokey) |> dplyr::filter(!is.na(sandtotal_r) | !is.na(silttotal_r)| !is.na(claytotal_r)| !is.na(fragvol_r)) |> dplyr::select(bottom) |> dplyr::filter(bottom == max(bottom))

  data_profile_list <- list(list())
  for(i in 1:nrow(max_depth)){
    data_profile_list[[i]] <- data_profile_slab |> dplyr::filter(cokey==max_depth$cokey[i]) |> dplyr::filter(bottom <= max_depth$bottom[i])
  }
  data_profile_std <-  dplyr::bind_rows(data_profile_list) |> as.data.frame()
  data_profile_std <- data_profile_std |> dplyr::mutate(silttotal_l = replace(silttotal_l, silttotal_l<0, 0), silttotal_h = replace(silttotal_h, silttotal_h<0, 0), sandtotal_l = replace(sandtotal_l, sandtotal_l<0, 0), sandtotal_h = replace(sandtotal_h, sandtotal_h<0, 0))
  data_profile_std[is.na(data_profile_std)] <- 0.01
  data_profile_std[data_profile_std == -9999.00000] <-  0.01
  data_profile_std[,3:15][data_profile_std[,3:15] == 0] <-  0.01

  data_profile_std$cokey <- as.integer(data_profile_std$cokey)
  data_profile_std <- data_profile_std |> dplyr::left_join(data |> dplyr::select(cokey, mukey, compname, restLyr, comppct_r, rep_n) |> dplyr::distinct(), by="cokey")

  # # simulate profile data
  # sim_soil <- mukey_sim(data_profile_std)
  #
  top <- intervals[1:(length(intervals)-1)]
  bottom <- intervals[2:length(intervals)]
  Depth <- list()
  for(i in 1:length(top)){
    Depth[i] <- paste0(top[i], "-", bottom[i], " cm")
  }
  depth_intervals <- data.frame(unlist(Depth), top) |> purrr::set_names("Depth", 'top')

  data_profile_std  <-  data_profile_std   |> dplyr::left_join(depth_intervals, by="top")
  data_profile_std  <-  data_profile_std  |> dplyr::group_by(compname, Depth) |> dplyr::mutate(id = dplyr::row_number()) |> dplyr::ungroup()
  if(properties == "ALL"){
    data_wide <- data_profile_std  |> dplyr::select(-c(top,bottom)) |> tidyr::pivot_wider(names_from = Depth, values_from = c(sandtotal_r, silttotal_r,claytotal_r,fragvol_r,ph1to1h2o_r,om_r,awc_r,dbovendry_r))
  } else if(properties == "LPKS"){
    data_wide <- sim_soil |> dplyr::select(-c(top,bottom)) |> tidyr::pivot_wider(names_from = Depth, values_from = c(sandtotal_r, silttotal_r,claytotal_r,fragvol_r))
  }
  data_wide[is.na(data_wide)] <- 0.01
  return(sim_data_wide)
}


# #Feature selection
# #weights <- information.gain(compname ~ ., data = sim_data_wide)
# weights <- FSelector::gain.ratio(compname ~ ., data = sim_data_wide |> dplyr::select(-c(mukey, cokey,id))) |> dplyr::arrange(desc(attr_importance))
# subset <- FSelector::cutoff.k(weights, 3)
#
# sim_data_wide <- sim_data_wide |> dplyr::mutate(compname=as.factor(compname))
# sim_data_wide <- sim_data_wide |> dplyr::filter(compname %in% c(comp_data$compname))
# sim_data_wide$compname <- droplevels(sim_data_wide$compname)
# comp_level <- levels(sim_data_wide$compname) |> as.data.frame() |> purrr::set_names('compname') |> dplyr::left_join(comp_data |> dplyr::select(compname, loc_score_comp) |> dplyr::distinct(), by="compname")
# ###########################################################################################
# #try bnlearn::tree.bayes
# set.seed(120, "L'Ecuyer-CMRG")
# #need to input prior probabilities based on distance weights --  |> dplyr::select(starts_with(subset))
# if(sub==TRUE){
#   mod <- naivebayes::naive_bayes(compname ~ ., sim_data_wide |> dplyr::select(compname, starts_with(subset)) |> as.data.frame(), laplace=1, na.action = stats::na.pass, prior = c(soilid_list$distance_score))
#   val <- data.frame(stats::predict(mod, newdata = val_data_wide |> dplyr::select(starts_with(subset)) |> as.data.frame() , type='prob')) |> t() |> as.data.frame()
#   val <- val  |> dplyr::mutate(compname=rownames(val)) |> purrr::set_names('prob' , 'compname') |> dplyr::select(compname, prob) |> dplyr::arrange(desc(prob))|> dplyr::mutate(prob=prob |> round(3))
# }else{
#   mod <- naivebayes::naive_bayes(compname ~ ., sim_data_wide |> dplyr::select(-c(mukey, cokey,id)) |> as.data.frame(), laplace=1, na.action = stats::na.pass, prior = c(comp_level$loc_score_comp))
#   val <- data.frame(stats::predict(mod, newdata = val_data_wide |> as.data.frame() , type='prob')) |> t() |> as.data.frame()
#   val <- val  |> dplyr::mutate(compname=rownames(val)) |> purrr::set_names('prob' , 'compname') |> dplyr::select(compname, prob) |> dplyr::arrange(desc(prob))|> dplyr::mutate(prob=prob |> round(3))
# }
#



### --------------------------------------------------------------------------------------





# OLD FUNCTIONS

#return list of cokey ids within 1000m of point -- this requires adjacency to home mapunit
soilweb_cokey_search <-  function(lat, lon){
  soilweb_url <- paste("https://casoilresource.lawr.ucdavis.edu/api/landPKS.php?q=spn&lon=", lon, "&lat=", lat, "&r=1", sep="")
  soilweb_return <- purrr::possibly(fromJSON, "Failed")(soilweb_url, simplifyDataFrame = TRUE)
  #get list of cokeys within search radius
  cokey_list <- soilweb_return$spn |> dplyr::select(cokey, distance_m)
  return(cokey_list)
}

SDA_mukey_return <-  function(lat, lon){
  mukey_return <- soilDB::SDA_query(paste0("SELECT mukey, muname
  FROM mapunit
  WHERE mukey IN (
  SELECT * from SDA_Get_Mukey_from_intersection_with_WktWgs84('point(", lon, " ", lat, ")') )"))
  return(mukey_return)
}

SDA_cokey_return <-  function(lat, lon){
  cokey_return <- soilDB::SDA_query(paste0("SELECT component.mukey, component.cokey,  component.compname
  FROM component
  WHERE component.mukey IN (
  SELECT * from SDA_Get_Mukey_from_intersection_with_WktWgs84('point(", lon, " ", lat, ")') )"))
  return(cokey_return)
}


# Property range data simulation - Naive Bayes depth-wise probability
# Function inputs: pedon = data.frame with required soil property data; sub = boolean (True/False) indicating if feature selection should be used to subset inputs
# nb_depth_sim <- function(pedon, sub){
#   # processing pedon data - remove horizons below bedrock depth, convert to standard depth intervals
#   pedon <- hrz_bdrock_clip(pedon)
#   pedon <- pedon |> dplyr::filter(!hzn_desgn %in% bedrock_hzname)
#   pedon <- pedon |> dplyr::filter(!is.na(hzdept) & !is.na(hzdepb)) |> dplyr::filter(!is.na(sand) | !is.na(silt)| !is.na(clay))
#   pedon <- pedon |> dplyr::mutate(compname = stringr::str_to_lower(taxonname))
#   if(all(is.na(pedon$bedrkdepth))==TRUE & max(pedon$hzdepb) >120){
#     max_D <- 120
#   } else if(all(is.na(pedon$bedrkdepth))==TRUE & max(pedon$hzdepb) <120) {
#     max_D <- max(pedon$hzdepb)
#   } else if(unique(pedon$bedrkdepth)>120 & max(pedon$hzdepb) >120) {
#     max_D <- 120
#   } else if(unique(pedon$bedrkdepth)>120 & max(pedon$hzdepb) <120) {
#     max_D <- max(pedon$hzdepb)
#   } else {
#     max_D <- unique(pedon$bedrkdepth) -1
#   }
#   pedon_spd <- pedon |> dplyr::select(compname, hzdept, hzdepb, hzn_desgn, sandtotal_r=sand, silttotal_r=silt, claytotal_r=clay, fragvol_r=rfv, rfvclass, textclass)
#   aqp::depths(pedon_spd) <- compname ~ hzdept + hzdepb
#   pedon_slab <- aqp::slab(pedon_spd, fm = compname ~ sandtotal_r + silttotal_r + claytotal_r + fragvol_r, slab.structure=c(0,10,20,30,50,70,100,120), slab.fun=mean_na)
#   pedon_slab <- pedon_slab |> dplyr::select(-c(contributing_fraction)) |> tidyr::pivot_wider(names_from = variable, values_from = value) |> dplyr::rename(hzdept=top, hzdepb=bottom)
#   pedon_slab[ is.na(pedon_slab) ] <- NA
#   max_depth <- pedon_slab |> dplyr::group_by(compname) |> dplyr::filter(!is.na(sandtotal_r) | !is.na(silttotal_r)| !is.na(claytotal_r)| !is.na(fragvol_r)) |> dplyr::select(hzdepb) |> dplyr::filter(hzdepb == max(hzdepb))
#   pedon_list <- list(list())
#   for(i in 1:nrow(max_depth)){
#     pedon_list[[i]] <- pedon_slab |> dplyr::filter(compname==max_depth$compname[i]) |> dplyr::filter(hzdepb <= max_depth$hzdepb[i])
#   }
#   pedon_std <-  dplyr::bind_rows(pedon_list)
#   for(i in 1:nrow(pedon_std))  {
#     pedon_std$textclass[i] <- gettt(pedon_std$sandtotal_r[i], pedon_std$silttotal_r[i], pedon_std$claytotal_r[i])
#     pedon_std$rfvclass[i] <- getCF_groups(pedon_std$fragvol_r[i])
#   }
#   pedon_std$soil_key <- unique(pedon$soil_key)
#   pedon_std$taxonname <- unique(pedon$taxonname)
#
#   pedon_std <- pedon_std |> dplyr::mutate(bedrkdepth=max_D)
#
#   #component data
#   #Query soilweb based on pedon lat/lon
#   lon=as.numeric(unique(pedon$lon))
#   lat=as.numeric(unique(pedon$lat))
#
#   reticulate::source_python("C:/R_Drive/Data_Files/LPKS_Data/R_Projects/SoilID.2.0/code/Global_soilid2_functions_p3.py")
#   soilid_list_all <- mucomp_out(lon, lat)
#   soilid_list <- soilid_list_all |> dplyr::select(compname_grp, distance_score) |> dplyr::group_by(compname_grp) |> dplyr::slice_max(distance_score) |> dplyr::ungroup() |> dplyr::distinct()
#
#   soilweb_url <- paste("https://casoilresource.lawr.ucdavis.edu/api/landPKS.php?q=spn&lon=", lon, "&lat=", lat, "&r=1000", sep="")
#   soilweb_return <- purrr::possibly(fromJSON, "Failed")(soilweb_url, simplifyDataFrame = TRUE)
#   #get list of cokeys within search radius
#   cokey_list <- soilweb_return$spn$cokey
#
#   # chorizon SDA query
#   muhorzdataQry =  paste('SELECT cokey,  chorizon.chkey, hzdept_r, hzdepb_r, hzname, sandtotal_l,sandtotal_h, sandtotal_r, silttotal_l, silttotal_h, silttotal_r, claytotal_l, claytotal_h, claytotal_r, chfrags.fragvol_l, chfrags.fragvol_h, chfrags.fragvol_r FROM chorizon LEFT OUTER JOIN chfrags ON chfrags.chkey = chorizon.chkey WHERE cokey IN (', paste(cokey_list, collapse=", "), ')', sep="")
#
#   horz_SDA = soilDB::SDA_query(muhorzdataQry)
#   horz_SDA <- horz_SDA  |> dplyr::group_by(cokey, hzdept_r) |> dplyr::mutate(fragvol_l = sum(fragvol_l)) |> dplyr::mutate(fragvol_h = sum(fragvol_h)) |> dplyr::mutate(fragvol_r = sum(fragvol_r)) |> dplyr::ungroup() |> dplyr::distinct()
#   ssurgo_profile = horz_SDA |> dplyr::inner_join(soilweb_return$spn |> dplyr::select(cokey, compname, comppct_r), by="cokey")
#   ssurgo_profile$taxonname <- unique(pedon$taxonname)
#
#   soilid_list <- soilid_list |> dplyr::filter(compname_grp %in% unique(ssurgo_profile$compname))
#
#   #Component restrictions query
#   mucomprestrdataQry =  paste('SELECT corestrictions.cokey, corestrictions.reskind, corestrictions.reshard, corestrictions.resdept_r, corestrictions.resdepb_r FROM corestrictions WHERE corestrictions.cokey IN (', paste(cokey_list, collapse=", "), ')', sep="")
#   comprestr_SDA <- soilDB::SDA_query(mucomprestrdataQry)
#   if(is.null(comprestr_SDA)){
#     ssurgo_profile$bedrkdepth <- NA
#   } else {
#     comprestr_filter <- comprestr_SDA |> dplyr::filter(reskind %in% c("Lithic bedrock", "Paralithic bedrock")) |> dplyr::group_by(cokey) |> dplyr::mutate(bedrkdepth = min(resdept_r)) |> dplyr::select(cokey, bedrkdepth) |> dplyr::ungroup()
#     ssurgo_profile <- ssurgo_profile |> dplyr::left_join(comprestr_filter, by="cokey")
#   }
#
#   ssurgo_profile <- ssurgo_profile |> dplyr::filter(!is.na(hzdept_r) & !is.na(hzdepb_r)) |> dplyr::filter(!is.na(sandtotal_r)  &  !is.na(silttotal_r) &  !is.na(claytotal_r))
#   ssurgo_profile <- ssurgo_profile |> dplyr::filter(!hzname %in% bedrock_hzname)
#
#   #Remove any pedons horizons below bedrockdepth
#   soil_key_list <- ssurgo_profile |> dplyr::select(cokey) |> dplyr::distinct() |> pull()
#   ssurgo_profile_lst <- list(list())
#   for(j in 1:length(soil_key_list)){
#     ped <- ssurgo_profile |> dplyr::filter(cokey==soil_key_list[j])
#     ped <- hrz_bdrock_clip(ped)
#     ssurgo_profile_lst[[j]] <-  ped
#   }
#   ssurgo_profile <-  dplyr::bind_rows(ssurgo_profile_lst)
#
#   ssurgo_profile$bedrkdepth <-  as.numeric(ssurgo_profile$bedrkdepth)
#   ssurgo_profile$hzdepb_r <-  as.numeric(ssurgo_profile$hzdepb_r)
#   ssurgo_profile$hzdept_r <-  as.numeric(ssurgo_profile$hzdept_r)
#
#   ssurgo_profile <-  ssurgo_profile |> dplyr::group_by(cokey) |> dplyr::mutate(bedrkdepth = case_when(
#     all(is.na(bedrkdepth))==TRUE & max(hzdepb_r) >120 ~ 120,
#     all(is.na(bedrkdepth))==TRUE & max(hzdepb_r) <120 ~ max(hzdepb_r),
#     unique(bedrkdepth)>120 & max(hzdepb_r) >120 ~ 120,
#     unique(bedrkdepth)>120 & max(hzdepb_r) <120 ~ max(hzdepb_r),
#     TRUE                           ~ unique(bedrkdepth)))  |> dplyr::ungroup()
#
#
#   ssurgo_profile_slab <- ssurgo_profile |> dplyr::arrange(hzdept_r) |> dplyr::arrange(cokey) |> data.frame()
#   aqp::depths(ssurgo_profile_slab) <-  cokey ~ hzdept_r + hzdepb_r
#   #ssurgo_profile_bedrock <- ssurgo_profile |> dplyr::select(compname, bedrkdepth) |> dplyr::mutate(bedrkdepth=replace_na(bedrkdepth, 120))
#
#   ssurgo_profile_slab <- aqp::slab(ssurgo_profile_slab, fm = cokey ~ sandtotal_l + sandtotal_h + sandtotal_r + silttotal_l + silttotal_h + silttotal_r + claytotal_l + claytotal_h + claytotal_r + fragvol_l + fragvol_h + fragvol_r, slab.structure=c(0,10,20,50,70,100,120), slab.fun=mean_na)
#   ssurgo_profile_slab <- ssurgo_profile_slab |> dplyr::select(-c(contributing_fraction)) |> spread(variable, value) |> dplyr::rename(hzdept=top, hzdepb=bottom)
#   ssurgo_profile_slab[ is.na(ssurgo_profile_slab) ] <- NA
#   max_depth <- ssurgo_profile_slab |> dplyr::group_by(cokey) |> dplyr::filter(!is.na(sandtotal_r) | !is.na(silttotal_r)| !is.na(claytotal_r)| !is.na(fragvol_r)) |> dplyr::select(hzdepb) |> filter(hzdepb == max(hzdepb))
#   ssurgo_profile_list <- list(list())
#   for(i in 1:nrow(max_depth)){
#     ssurgo_profile_list[[i]] <- ssurgo_profile_slab |> dplyr::filter(cokey==max_depth$cokey[i]) |> dplyr::filter(hzdepb <= max_depth$hzdepb[i])
#   }
#   ssurgo_profile_std <-  dplyr::bind_rows(ssurgo_profile_list)
#
#   depth_intervals <- data.frame(c('0-10 cm', '10-20 cm', '20-50 cm', '50-70 cm', '70-100 cm', '100-120 cm'),c(0, 10, 20, 50, 70, 100)) |> set_names("Depth", 'hzdept')
#   ssurgo_profile_std <-  ssurgo_profile_std  |> dplyr::left_join(depth_intervals, by="hzdept")
#   ssurgo_profile_compname <- ssurgo_profile |> dplyr::select(cokey, compname) |> dplyr::distinct()
#   ssurgo_profile_std$cokey <- as.integer(ssurgo_profile_std$cokey)
#   ssurgo_profile_std <- ssurgo_profile_std |> dplyr::left_join(ssurgo_profile_compname, by="cokey")
#   ssurgo_profile_std <- ssurgo_profile_std |> dplyr::left_join(ssurgo_profile |> dplyr::select(cokey, bedrkdepth) |> dplyr::distinct(), by="cokey")
#   ssurgo_profile_std <- ssurgo_profile_std |> dplyr::group_by(cokey) |> dplyr::mutate(layer = seq(1, length(cokey), 1))  |> dplyr::ungroup()  |> as.data.frame()
#   ssurgo_profile_std <- ssurgo_profile_std |> dplyr::select(compname, Depth, id=cokey, sandtotal_r, sandtotal_l, sandtotal_h, silttotal_r, silttotal_l, silttotal_h, claytotal_r, claytotal_l, claytotal_h, fragvol_r, fragvol_l, fragvol_h, layer, bedrkdepth, hzdept_r=hzdept, hzdepb_r=hzdepb, layer)
#   ssurgo_profile_std  <- ssurgo_profile_std  |> dplyr::group_by(compname, Depth) |> dplyr::summarise_all(funs(mean(.,na.rm = T))) |> dplyr::ungroup() |> dplyr::arrange(id,layer)
#
#   #Code to infill l/h values for silt/sand when clay values are complete
#   for(l in 1:nrow(ssurgo_profile_std))
#     if(is.na(ssurgo_profile_std$sandtotal_l[l]) & is.na(ssurgo_profile_std$sandtotal_h[l]) & !is.na(ssurgo_profile_std$claytotal_l[l])& !is.na(ssurgo_profile_std$claytotal_h[l])){
#       ssurgo_profile_std$sandtotal_l[l] <- 100 - (ssurgo_profile_std$silttotal_r[l] + ssurgo_profile_std$claytotal_h[l])
#       ssurgo_profile_std$sandtotal_h[l] <- 100 - (ssurgo_profile_std$silttotal_r[l] + ssurgo_profile_std$claytotal_l[l])
#     }
#   for(m in 1:nrow(ssurgo_profile_std))
#     if(is.na(ssurgo_profile_std$silttotal_l[m]) & is.na(ssurgo_profile_std$silttotal_h[m]) & !is.na(ssurgo_profile_std$claytotal_l[m])& !is.na(ssurgo_profile_std$claytotal_h[m])){
#       ssurgo_profile_std$silttotal_l[m] <- 100 - (ssurgo_profile_std$sandtotal_r[m] + ssurgo_profile_std$claytotal_h[m])
#       ssurgo_profile_std$silttotal_h[m] <- 100 - (ssurgo_profile_std$sandtotal_r[m] + ssurgo_profile_std$claytotal_l[m])
#     }
#   ssurgo_profile_std <- ssurgo_profile_std |> dplyr::mutate(silttotal_l = replace(silttotal_l, silttotal_l<0, 0), silttotal_h = replace(silttotal_h, silttotal_h<0, 0), sandtotal_l = replace(sandtotal_l, sandtotal_l<0, 0), sandtotal_h = replace(sandtotal_h, sandtotal_h<0, 0))
#   ssurgo_profile_std <- ssurgo_profile_std |> replace(., is.na(.), 0.1)
#   ssurgo_profile_std[ssurgo_profile_std == -9999.00000] <-  0.1
#   ssurgo_profile_std[,3:15][ssurgo_profile_std[,3:15] == 0] <-  0.1
#
#
#   sim_texture_list <- list(list())
#   for(j in 1:nrow(ssurgo_profile_std)){
#     sim_texture_list[[j]] <- text_sim(ssurgo_profile_std[j,])
#   }
#
#   sim_texture <- dplyr::bind_rows(sim_texture_list)
#   sim_texture <- sim_texture |> dplyr::select(compname, Depth, sandtotal_r, silttotal_r, claytotal_r,fragvol_r, bottom) |> dplyr::mutate(compname = as.factor(compname))
#   sim_texture <-  sim_texture |> dplyr::group_by(compname, Depth) |> dplyr::mutate(id = row_number()) |> dplyr::ungroup()
#
#   #levels(pedon_val$compname) <- levels(sim_texture$compname)
#   pedon_val <-  pedon_val |> dplyr::mutate(bedrkdepth=replace(bedrkdepth, bedrkdepth>120, 120))
#
#   #Model across all depths
#   sim_data_wide <- sim_texture |> dplyr::select(-c(bottom)) |> tidyr::pivot_wider(names_from = Depth, values_from = c(sandtotal_r, silttotal_r,claytotal_r,fragvol_r)) |> dplyr::select(-c(id))
#   sim_data_wide <- sim_data_wide |> replace(., is.na(.), 0)
#   # sim_data_wide <-  sim_data_wide |> dplyr::left_join(ssurgo_profile_bedrock, by="compname")
#   # sim_data_wide <-   sim_data_wide |> dplyr::mutate(bedrkdepth = as.factor(bedrkdepth))
#
#   pedon_val <- pedon_val |> rowwise() |> dplyr::mutate(sand=getSand(gettt(sandtotal_r,silttotal_r,claytotal_r)),clay = getClay(gettt(sandtotal_r,silttotal_r,claytotal_r)), silt= 100-(sand+clay), rfv=getCF(fragvol_r) |> as.numeric()) |> dplyr::ungroup() |> dplyr::mutate(silt = if_else(silt >99, 0.01, silt))
#   pedon_val <-  pedon_val  |> dplyr::left_join(depth_intervals, by="hzdept")
#
#   val_data_wide <- pedon_val  |> dplyr::select(c(sandtotal_r=sand, silttotal_r=silt,claytotal_r=clay,fragvol_r=rfv,  Depth)) |> tidyr::pivot_wider(names_from = Depth, values_from = c(sandtotal_r, silttotal_r,claytotal_r,fragvol_r))
#   val_data_wide <- val_data_wide |> replace(., is.na(.), 0.01)
#   val_data_wide[val_data_wide == 0] <-  0.01
#   # val_data_wide <-   val_data_wide |> dplyr::mutate(bedrkdepth = unique(pedon_val$bedrkdepth))|> dplyr::mutate(bedrkdepth = as.factor(bedrkdepth))
#
#   #Feature selection
#   #weights <- information.gain(compname ~ ., data = sim_data_wide)
#   weights <- FSelector::gain.ratio(compname ~ ., data = sim_data_wide) |> dplyr::arrange(desc(attr_importance))
#   subset <- FSelector::cutoff.k(weights, 3)
#
#   sim_data_wide <- sim_data_wide |> dplyr::mutate(compname=as.factor(compname))
#   sim_data_wide <- sim_data_wide |> dplyr::filter(compname %in% c(soilid_list$compname_grp))
#   sim_data_wide$compname <- droplevels(sim_data_wide$compname)
#
#   ###########################################################################################
#   set.seed(120, "L'Ecuyer-CMRG")
#   #need to input prior probabilities based on distance weights --  |> dplyr::select(starts_with(subset))
#   if(sub==TRUE){
#     mod <- naivebayes::naive_bayes(compname ~ ., sim_data_wide |> dplyr::select(compname, starts_with(subset)) |> as.data.frame(), laplace=1, na.action = stats::na.pass, prior = c(soilid_list$distance_score))
#     val <- data.frame(stats::predict(mod, newdata = val_data_wide |> dplyr::select(starts_with(subset)) |> as.data.frame() , type='prob')) |> t() |> as.data.frame()
#     val <- val  |> dplyr::mutate(compname=rownames(val)) |> purrr::set_names('prob' , 'compname') |> dplyr::select(compname, prob) |> dplyr::arrange(desc(prob))|> dplyr::mutate(prob=prob |> round(3))
#   }else{
#     mod <- naivebayes::naive_bayes(compname ~ ., sim_data_wide |> as.data.frame(), laplace=1, na.action = stats::na.pass, prior = c(soilid_list$distance_score))
#     val <- data.frame(stats::predict(mod, newdata = val_data_wide |> as.data.frame() , type='prob')) |> t() |> as.data.frame()
#     val <- val  |> dplyr::mutate(compname=rownames(val)) |> purrr::set_names('prob' , 'compname') |> dplyr::select(compname, prob) |> dplyr::arrange(desc(prob))|> dplyr::mutate(prob=prob |> round(3))
#   }
#   return(val)
# }

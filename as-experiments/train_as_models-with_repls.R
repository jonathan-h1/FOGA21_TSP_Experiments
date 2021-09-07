library(batchtools)
library(tidyverse)
library(BBmisc)
library(checkmate)
library(salesperson)
library(mlr)

library(parallel)
library(ParamHelpers)
library(kernlab)
library(xgboost)
library(randomForest)
library(rpart)

##################################################################################################

## create registry
unlink("AS-Experiments-TSP-with_Repls", recursive = TRUE)
reg = batchtools::makeExperimentRegistry(file.dir = "AS-Experiments-TSP-with_Repls",
  packages = c("tidyverse", "BBmisc", "checkmate", "mlr", "salesperson", "parallel", "ParamHelpers", "kernlab", "randomForest", "xgboost", "rpart"))

##################################################################################################

preprocessASData = function(root.sets = "ecj2018", feature.sets = "all", tsp.sets = "all", solver.set = c("eax", "lkh"), perf.measure = "pqr10", normalized.feats = TRUE, wd = getwd()) {

  ## check that the feature, tsp and solver sets are characters
  assertCharacter(x = root.sets)
  root.sets = strsplit(root.sets, "[|]")[[1L]]
  assertCharacter(x = feature.sets)
  feature.sets = strsplit(feature.sets, "[|]")[[1L]]
  assertCharacter(x = tsp.sets)
  tsp.sets = strsplit(tsp.sets, "[|]")[[1L]]
  assertCharacter(x = solver.set)
  solver.set = strsplit(solver.set, "[|]")[[1L]]
  assertChoice(x = perf.measure, choices = c("par10", "pqr10"), null.ok = FALSE)

  ## instance identifier
  inst.id = c("root", "group", "prob")

  ## overview of all valid names
  all.feats = c("angles_angle", "angles_cos", "bounding_box_01pct", "bounding_box_02pct", "bounding_box_03pct", "centroid",
    "cluster_01pct", "cluster_05pct", "cluster_10pct", "hull_points", "hull_area", "hull_edges", "hull_dists",
    "distance", "modes", "mst_depth", "mst_dists", "nearest_neighbour", "nng", "size", "autoencoder")
  all.sets = c("national-500-5000", "netgen", "netgen_morphed", "rue", "tsplib", "vlsi-500-5000",
    "eax---lkh---simple", "eax---lkh---sophisticated", "lkh---eax---simple", "lkh---eax---sophisticated", "tspgen")
  all.roots = c("ecj2018", "evolved500", "evolved1000", "evolved1500", "evolved2000", "foga2019")
  all.solver = c("eax", "lkh", "concorde")
  
  ## define feature, tsp and solver sets
  if ("all" %in% feature.sets) {
    feature.sets = all.feats
  }
  if ("all" %in% tsp.sets) {
    if ("all" %in% root.sets) {
      tsp.sets = all.sets
    } else {
      sets = NULL
      if ("foga2019" %in% root.sets) {
        sets = c(sets, "tspgen")
      }
      if (any(grepl(pattern = "evolved", x = root.sets))) {
        sets = c(sets, c("eax---lkh---simple", "eax---lkh---sophisticated", "lkh---eax---simple", "lkh---eax---sophisticated"))
      }
      if ("ecj2018" %in% root.sets) {
        sets = c(sets, c("national-500-5000", "netgen", "netgen_morphed", "rue", "tsplib", "vlsi-500-5000"))
      }
      tsp.sets = sets
    }
  }
  if ("all" %in% root.sets) {
    root.sets = all.roots
  }
  if ("all" %in% solver.set) {
    solver.set = all.solver
  }
  
  ## check whether root, feature, tsp and solver sets only contain valid strings
  if (length(setdiff(root.sets, all.roots)) > 0L) {
    BBmisc::stopf("The 'root.sets' contain the following unknown sets: %s", paste(setdiff(root.sets, all.roots), collapse = ", "))
  }
  if (length(setdiff(feature.sets, all.feats)) > 0L) {
    BBmisc::stopf("The 'feature.sets' contain the following unknown sets: %s", paste(setdiff(feature.sets, all.feats), collapse = ", "))
  }
  if (length(setdiff(tsp.sets, all.sets)) > 0L) {
    BBmisc::stopf("The 'tsp.sets' contain the following unknown sets: %s", paste(setdiff(tsp.sets, all.sets), collapse = ", "))
  }
  if (length(setdiff(solver.set, all.solver)) > 0L) {
    BBmisc::stopf("The 'solver.set' contains the following unknown solvers: %s", paste(setdiff(solver.set, all.solver), collapse = ", "))
  }

  #### Preprocess Performance Data
  ## import performance data
  perfs = readr::read_csv(file = "../tsp-instances/performance data/combined.csv")
  perf.columns = sprintf("root|group|prob|fold|%s", paste(sprintf("%s.%s$", solver.set, perf.measure), collapse = "|"))
  perfs = perfs[, grepl(perf.columns, colnames(perfs))]
  
  ## filter for relevant instances
  perfs = perfs %>% dplyr::filter(root %in% root.sets, group %in% tsp.sets)

  ## ensure that folds are of type integer
  perfs = perfs %>% dplyr::mutate_at("fold", as.integer)

  ## remove the name of the performance measure
  colnames(perfs) = gsub(pattern = sprintf(".%s", perf.measure), replacement = "", colnames(perfs))


  #### Preprocess Feature Data
  ## import feature data
  feats = readr::read_csv(file = "../tsp-instances/feature sets/feature_set_nan.csv")

  ## import autoencoder feats
  suppressWarnings((feats.ae.norm = readr::read_csv(file = "../tsp-instances/feature sets/autoencoder_normed_combined.csv")[,-1L]))
  feats.ae.norm$norm = TRUE
  suppressWarnings((feats.ae.unnorm = readr::read_csv(file = "../tsp-instances/feature sets/autoencoder_unnormed_combined.csv")[,-1L]))
  feats.ae.unnorm$norm = FALSE
  feats.autoencoder = bind_rows(feats.ae.norm, feats.ae.unnorm) %>% 
    rename_with(function(x) gsub(pattern = "X", replacement = "autoencoder_X", x = x), starts_with("X"))

  ## add problem size as feature
  feats$size = NA_integer_
  index = feats$group %in% c("eax---lkh---simple", "eax---lkh---sophisticated", "lkh---eax---simple", "lkh---eax---sophisticated", "netgen", "netgen_morphed", "rue", "tspgen")
  feats$size[index] = vapply(strsplit(feats$prob[index], "-"), function(x) as.integer(x[1L]), integer(1L))
  x = feats$prob[!index]
  x = gsub(pattern = ".tsp", replacement = "", x)
  x = gsub(pattern = paste(c(letters, LETTERS), collapse = "|"), replacement = "", x)
  feats$size[!index] = vapply(x, as.integer, integer(1L))
  if (any(grepl("lu980", feats$prob))) {
    feats$size[grepl("lu980", feats$prob)] = 634L
  }
  if (any(grepl("rw1621", feats$prob))) {
    feats$size[grepl("rw1621", feats$prob)] = 866L
  }
  feats$size_costs = 0L

  ## fix naming issues in feature names
  feats = feats %>% 
    rename_with(function(x) gsub(pattern = "angle_", replacement = "angles_", x = x), starts_with("angle_")) %>%
    rename_with(function(x) gsub(pattern = "angles_cos_", replacement = "cos_angles_", x = x), starts_with("angles_cos_")) %>%
    rename_with(function(x) gsub(pattern = "angles_", replacement = "angles_angle_", x = x), starts_with("angles_")) %>%
    rename_with(function(x) gsub(pattern = "cos_angles_", replacement = "angles_cos_", x = x), starts_with("cos_angles_")) %>%
    rename_with(function(x) gsub(pattern = "fraction_of_nodes_outside_near_bounding_box_0.1", replacement = "bounding_box_01pct_fraction_of_nodes_outside_near", x = x),
      matches("fraction_of_nodes_outside_near_bounding_box_0.1")) %>%
    rename_with(function(x) gsub(pattern = "fraction_of_nodes_outside_near_bounding_box_0.2", replacement = "bounding_box_02pct_fraction_of_nodes_outside_near", x = x),
      matches("fraction_of_nodes_outside_near_bounding_box_0.2")) %>%
    rename_with(function(x) gsub(pattern = "fraction_of_nodes_outside_near_bounding_box_0.3", replacement = "bounding_box_03pct_fraction_of_nodes_outside_near", x = x),
      matches("fraction_of_nodes_outside_near_bounding_box_0.3")) %>%
    rename_with(function(x) gsub(pattern = "bounding_box_0.1_costs", replacement = "bounding_box_01pct_costs", x = x), matches("bounding_box_0.1_costs")) %>%
    rename_with(function(x) gsub(pattern = "bounding_box_0.2_costs", replacement = "bounding_box_02pct_costs", x = x), matches("bounding_box_0.2_costs")) %>%
    rename_with(function(x) gsub(pattern = "bounding_box_0.3_costs", replacement = "bounding_box_03pct_costs", x = x), matches("bounding_box_0.3_costs")) %>%
    rename_with(function(x) gsub(pattern = "cluster_0.01_costs", replacement = "cluster_01pct_costs", x = x), matches("cluster_0.01_costs")) %>%
    rename_with(function(x) gsub(pattern = "cluster_0.05_costs", replacement = "cluster_05pct_costs", x = x), matches("cluster_0.05_costs")) %>%
    rename_with(function(x) gsub(pattern = "cluster_0.1_costs", replacement = "cluster_10pct_costs", x = x), matches("cluster_0.1_costs")) %>%
    rename_with(function(x) gsub(pattern = "fraction_shorter_mean_distance", replacement = "distance_fraction_shorter_mean_distance", x = x), matches("fraction_shorter_mean_distance")) %>%
    rename_with(function(x) gsub(pattern = "fraction_of_distinct_distances", replacement = "distance_fraction_of_distinct_distances", x = x), matches("fraction_of_distinct_distances")) %>%
    rename_with(function(x) gsub(pattern = "mode_frequency", replacement = "distance_mode_frequency", x = x), matches("mode_frequency")) %>%
    rename_with(function(x) gsub(pattern = "mode_quantity", replacement = "distance_mode_quantity", x = x), matches("mode_quantity")) %>%
    rename_with(function(x) gsub(pattern = "mode_mean", replacement = "distance_mode_mean", x = x), matches("mode_mean")) %>%
    rename_with(function(x) gsub(pattern = "mean_tour_length", replacement = "distance_mean_tour_length", x = x), matches("mean_tour_length")) %>%
    rename_with(function(x) gsub(pattern = "sum_of_lowest_edge_values", replacement = "distance_sum_of_lowest_edge_values", x = x), matches("sum_of_lowest_edge_values")) %>%
    rename_with(function(x) gsub(pattern = "nearest_neighbor_costs", replacement = "nearest_neighbour_costs", x = x), matches("nearest_neighbor_costs"))

  ## split MST costs into costs for the different MST streams
  feats = feats %>%
    mutate(
      mst_initialization_costs = mst_costs,
      mst_depth_costs = 0,
      mst_dists_costs = 0
    ) %>%
    select(- mst_costs)

  ## join regular features and autoencoder features (if autoencoder features are relevant for the analysis)
  if ("autoencoder" %in% feature.sets) {
    feats = feats %>% 
      left_join(feats.autoencoder, by = c("group", "prob", "norm")) %>%
      mutate(autoencoder_costs = 0L)
  }

  ## filter for relevant instances and keep only features that are part of one of the provided feature.sets
  feats = feats %>% 
    filter(root %in% root.sets, group %in% tsp.sets, norm == normalized.feats) %>%
    select(matches(c("root", "prob", "group")) | starts_with(feature.sets) | contains("initialization_costs"))

  ## remove features that compute the variance for a feature set
  feats = feats %>% select(!ends_with("_var"))
  
  fns = colnames(feats)
  for (fn in feature.sets) {
    if (any(grepl(fn, fns))) {
      fns = gsub(pattern = sprintf("%s_", fn), replacement = sprintf("%s__", fn), x = fns)
    }
  }
  if ("hull_area" %in% fns) {
    fns[fns == "hull_area"] = "hull_area__area"
  }
  if ("hull_initialization_costs" %in% fns) {
    fns[fns == "hull_initialization_costs"] = "hull_initialization__costs"
  }
  if ("angles_angle_initialization_costs" %in% fns) {
    fns[fns == "angles_angle_initialization_costs"] = "angles_initialization__costs"
  }
  if ("mst_initialization_costs" %in% fns) {
    fns[fns == "mst_initialization_costs"] = "mst_initialization__costs"
  }
  colnames(feats) = fns

  ## split feature data into feature and cost data
  cost.names = fns[grepl("_costs", fns)]
  costs = feats %>%
    select(all_of(c("root", "group", "prob", cost.names)))
  feats = feats %>%
    dplyr::select(!any_of(cost.names))

  ## remove features that are exclusively NAs
  feats = feats %>%
    select(where(~ !all(is.na(.x))))

  ## now also remove features, which contain some NAs
  feats = feats %>%
    select(where(~ all(!is.na(.x))))

  ## remove initialization costs for all feature sets which have not been considered
  fns = colnames(feats)  
  if (!any(grepl("angle", fns))) {
    costs = costs %>% select(-angles_initialization__costs)
  }
  if (!any(grepl("hull", fns))) {
    costs = costs %>% select(-hull_initialization__costs)
  }
  if (!any(grepl("mst", fns))) {
    costs = costs %>% select(-mst_initialization__costs)
  }

  #### Ensure same instances and same order across all data sets
  common.instances = dplyr::inner_join(x = perfs[, inst.id], y = feats[, inst.id], by = inst.id)
  common.instances = dplyr::inner_join(x = common.instances, y = costs[, inst.id], by = inst.id)
  costs = dplyr::left_join(x = common.instances, y = costs, by = inst.id)
  feats = dplyr::left_join(x = common.instances, y = feats, by = inst.id)
  perfs = dplyr::left_join(x = common.instances, y = perfs, by = inst.id)

  ## store separate data set with the folds
  folds = perfs %>% 
    select(all_of(c(inst.id, "fold")))

  ## split data into training and test data
  no.of.folds = max(folds$fold)
  rdesc = makeResampleDesc("CV", iters = no.of.folds)
  rinst = makeResampleInstance(desc = rdesc, size = nrow(perfs))
  ch = split(seq_row(folds), folds$fold)
  names(ch) = NULL
  rinst$test.inds = ch
  rinst$train.inds = lapply(ch, function(x) setdiff(BBmisc::seq_row(folds), x))
  
  ## remove features, which are constant within a fold of the training set
  ## (and therefore could lead to problems for the learning algorithms)
  feats = mlr::removeConstantFeatures(obj = feats, dont.rm = inst.id, na.ignore = FALSE)
  splitted.feats = lapply(rinst$train.inds, function(obs) feats[obs, setdiff(colnames(feats), inst.id), drop = FALSE])
  const.feats.within.fold = lapply(splitted.feats, function(split) {
    split %>% 
      select(where(~ min(.x, na.rm = TRUE) == max(.x, na.rm = TRUE))) %>% 
      names()
  })
  const.feats.within.fold = Reduce(union, const.feats.within.fold)
  const.feats.within.fold = setdiff(const.feats.within.fold, inst.id)
  feats = feats %>% 
    dplyr::select(!any_of(const.feats.within.fold))

  #### define measures (they are identical for multiple learning approaches)
  runtimes.exp.fun = function(task, model, pred, feats, extra.args) {
    classes = as.character(pred$data$response)
    ids = pred$data$id
    costs = task$costs
    ## Which feature sets are part of the current task?
    fns = getTaskFeatureNames(task)
    ## costs for considering the currently used feature sets
    if (length(fns) == 0) {
      used.featset.costs = rep(0, nrow(costs))
    } else {
      featset.costs = task$featset.costs
      cns = colnames(featset.costs)
      cns.wo.costs = gsub(pattern = "__costs", replacement = "", x = cns)
      used.featsets = vapply(cns.wo.costs, function(x) any(grepl(x, fns)), logical(1L))
      if (any(grepl("initialization", names(used.featsets))) && any(used.featsets)) {
        fs.names = names(used.featsets)
        init.index = grep("initialization", fs.names)
        used.featsets[init.index] = vapply(fs.names[init.index], function(init.ft) {
          any(used.featsets[grepl(gsub(x = init.ft, pattern = "_initialization", replacement = ""), fs.names)])
        }, logical(1L))
      }
      used.featset.costs = as.numeric(rowSums(featset.costs[, used.featsets, drop = FALSE]))
    }
    y = mapply(function(id, cl) {
      as.numeric(costs[id, cl]) + used.featset.costs[id]
    }, ids, classes, SIMPLIFY = TRUE, USE.NAMES = FALSE)
    mean(y)
  }
  runtimes.cheap.fun = function(task, model, pred, feats, extra.args) {
    classes = as.character(pred$data$response)
    ids = pred$data$id
    costs = task$costs
    y = mapply(function(id, cl) {
      as.numeric(costs[id, cl])
    }, ids, classes, SIMPLIFY = TRUE, USE.NAMES = FALSE)
    mean(y)
  }
  runtimes.cheap = makeMeasure(
    id = "runtimes.cheap", name = "Runtime Costs",
    properties = c("classif", "classif.multi", "req.pred", "costsens", "req.task"), minimize = TRUE,
    fun = runtimes.cheap.fun
  )
  runtimes.exp = makeMeasure(
    id = "runtimes.exp", name = "Runtime Costs with Costs for Feature Computation",
    properties = c("classif", "classif.multi", "req.pred", "costsens", "req.task"), minimize = TRUE,
    fun = runtimes.exp.fun
  )

  #############################################################################

  ## compute the par10-score for each solver and fold
  X = subset(perfs, select = solver.set)
  foldwise.perf.train = t(vapply(rinst$train.inds, function(ids) {
    colMeans(X[ids, , drop = FALSE])
  }, double(length(solver.set))))
  
  foldwise.perf.test = t(vapply(rinst$test.inds, function(ids) {
    colMeans(X[ids, , drop = FALSE])
  }, double(length(solver.set))))
  
  ## compute the vbs (needs to be done per fold as the fold size influences the results)
  vbs = mean(vapply(rinst$test.inds, function(inds) {
    test = X[inds, , drop = FALSE]
    mean(apply(test, 1, min))
  }, double(1L)))
  
  ## compute two versions of the sbs
  ## (1) sbs: first aggregate across all folds, then pick smallest runtime
  ## (2) sbs.cv: pick smallest runtime per fold and aggregated afterwards
  train.perf = colMeans(foldwise.perf.train)
  sbs = mean(foldwise.perf.test[, which(train.perf == min(train.perf))])
  sbs.cv = mean(vapply(seq_len(no.of.folds), function(j) {
    train.perf.fold = colMeans(foldwise.perf.train[j, , drop = FALSE])
    k = which(train.perf.fold == min(train.perf.fold))
    foldwise.perf.test[j, k]
  }, double(1L)))

  #############################################################################

  ## find the best solver per instance; if multiple ones exist, sample one of them
  sample.counter = 0L
  best.solver = vapply(BBmisc::seq_row(perfs), function(i) {
    pfs = perfs[i, solver.set]
    relevant.solver = solver.set[pfs == min(pfs)]
    if (length(relevant.solver) == 1L) {
      return(relevant.solver)
    } else {
      sample.counter <<- sample.counter + 1L
      return(sample(relevant.solver, 1L))
    }
  }, character(1L))

  if (sample.counter > 0L) {
    BBmisc::warningf("the best solver had to be sampled for %i instance%s",
      sample.counter, ifelse(sample.counter == 1L, "", "s"))
  }

  #############################################################################

  return(list(aggr.runtimes = perfs, costs = costs, feats = feats, rinst = rinst, best.solver = best.solver,
    # feature.sets.short = feature.sets.short, solver.sets.short = solver.sets.short, tsp.sets.short = tsp.sets.short,
    runtimes.cheap = runtimes.cheap, runtimes.exp = runtimes.exp, #filename = filename,
    feature.sets = feature.sets, tsp.sets = tsp.sets, solver.set = solver.set, root.sets = root.sets,
    foldwise.perf.test = foldwise.perf.test, foldwise.perf.train = foldwise.perf.train,
    vbs = vbs, sbs = sbs, sbs.cv = sbs.cv, wd = wd, inst.id = inst.id))
}

##################################################################################################

generateProblem = function(data, job, root.sets = "ecj2018", feature.sets = "all", tsp.sets = "all",
  solver.set = "eax|lkh", perf.measure = "pqr10", normalized.feats = TRUE, predict.type = "prob", learner, wd = getwd()) {

  job.name = sprintf("%s--%s--%s--%s--%s--%s--%s--%s",
    predict.type, paste(root.sets, collapse = "_"), paste(tsp.sets, collapse = "_"), paste(solver.set, collapse = "_"),
    paste(feature.sets, collapse = "_"), perf.measure, learner, as.character(normalized.feats))
  ## produce preprocessed problem instance
  prob.instance = preprocessASData(
    root.sets = root.sets, feature.sets = feature.sets, tsp.sets = tsp.sets, 
    solver.set = solver.set, perf.measure = perf.measure, normalized.feats = normalized.feats, wd = wd)

  ## convert feature data into a mlr-task
  df = bind_cols(prob.instance$feats, data.frame(solver = prob.instance$best.solver)) %>%
    select(!any_of(c("root", "prob", "group")))

  ## xgboost can't handle data that only consists of integer-values
  if (learner == "xgboost") {
    df = df %>% 
      mutate_if(is.integer, as.numeric)
  }

  task = makeClassifTask(id = "featTask", data = as.data.frame(df), target = "solver")

  ## add the runtimes and costs for the feature sets to the task
  ## (required for the computation of the exact runtime costs)
  task$costs = as.data.frame(prob.instance$aggr.runtimes %>% select(all_of(prob.instance$solver.set)))
  task$featset.costs = as.data.frame(prob.instance$costs %>% select(!any_of(prob.instance$inst.id)))

  ## in case of a SVM, estimate the parameter for sigma first
  if (learner == "ksvm") {
    sig = kernlab::sigest(solver ~ ., data = df)
    lrn = makeLearner("classif.ksvm", par.vals = list(sigma = sig[2L]), predict.type = predict.type)
  } else if (learner == "rf") {
    lrn = makeLearner("classif.randomForest", par.vals = list(ntree = 1L, replace = FALSE), predict.type = predict.type)
  } else if (learner == "xgboost") {
    lrn = makeLearner("classif.xgboost", par.vals = list(eval_metric = "logloss"), predict.type = predict.type)
  } else if (learner == "xgb") {
    lrn = makeLearner("classif.xgboost", par.vals = list(eval_metric = "error"), predict.type = predict.type)
  } else {
    lrn = makeLearner(sprintf("classif.%s", learner), predict.type = predict.type)
  }

  return(list(task = task, prob.instance = prob.instance, learner = lrn, job.name = job.name, wd = wd))
}

##################################################################################################

selectFeats = function(task, learner, fs.ctrl, s, repl, measure.list, resampling.instance, parallelize = TRUE) {
  n.cpus = parallel::detectCores()
  ## perform threshold-agnostic feature selection
  if (parallelize) {
    n.cpus = parallel::detectCores()
    parallelMap::parallelStartMulticore(cpus = n.cpus, level = "mlr.selectFeatures", show.info = FALSE)
  }
  set.seed(s + repl - 1L)
  sf = selectFeatures(learner = learner, task = task, measures = measure.list, resampling = resampling.instance,
    control = fs.ctrl, show.info = TRUE)
  if (parallelize) {
    parallelMap::parallelStop()
  }
  return(sf)
}

featselNone = function(data, job, instance, s, parallelize = TRUE) {
  res = classif(instance = instance, fs.ctrl = NULL, s = s, repl = job$repl, parallelize = parallelize)
  return(res)
}

featselSeq = function(data, job, instance, s, maxit = NA_integer_, max.features = NA_integer_, meth = "sffs", alpha = 0.0001, beta = 0, parallelize = TRUE) {
  fs.ctrl = makeFeatSelControlSequential(same.resampling.instance = TRUE, method = meth, alpha = alpha, beta = beta, max.features = max.features, maxit = maxit)
  res = classif(instance = instance, fs.ctrl = fs.ctrl, s = s, repl = job$repl, parallelize = parallelize)
  return(res)
}

featselExh = function(data, job, instance, s, maxit = NA_integer_, max.features = NA_integer_, parallelize = TRUE) {
  fs.ctrl = makeFeatSelControlExhaustive(same.resampling.instance = TRUE, max.features = max.features, maxit = maxit)
  res = classif(instance = instance, fs.ctrl = fs.ctrl, s = s, repl = job$repl, parallelize = parallelize)
  return(res)
}

classif = function(instance, fs.ctrl, s, repl, parallelize = TRUE) {
  if (!is.null(fs.ctrl)) {
    if (inherits(fs.ctrl, "FeatSelControlExhaustive")) {
      meth = sprintf("Exhaustive_%02i", mlr::getTaskNFeats(instance$task))
    } else {
      meth = gsub(pattern = "FeatSelControl", replacement = "", x = class(fs.ctrl)[1L])
      if (length(fs.ctrl$extra.args) == 0) {
        pars = ""
      } else {
        pars = paste(sprintf("%s_%s", names(unlist(fs.ctrl$extra.args)), unlist(fs.ctrl$extra.args)), collapse = "__")
      }
      # meth = paste(sprintf("%s_%s_%s", meth, names(unlist(fs.ctrl$extra.args)), unlist(fs.ctrl$extra.args)), collapse = "__")
      meth = sprintf("%s__%s", meth, pars)
    }
  } else {
    meth = "none"
  }
  lrn = instance$learner
  task = instance$task
  job.name = instance$job.name
  instance = instance$prob.instance
  resampling.instance = instance$rinst
  measure.list = list(instance$runtimes.exp, instance$runtimes.cheap)

  ## reduce task via feature selection
  if (meth != "none") {
    cat("Performing feature selection ...\n")
    sf = selectFeats(task = task, learner = lrn, fs.ctrl = fs.ctrl, s = s, repl = repl,
      measure.list = measure.list, resampling.instance = resampling.instance, parallelize = parallelize)
    feats = sf$x
    filename = sprintf("%s/intermediate_as_results/fs_with_repls-ex12/fs_classif--%s--%s--%04i--r%02i.rds", instance$wd, job.name, meth, s, repl)
    readr::write_rds(x = feats, file = filename)
    task = subsetTask(task = task, features = feats)
  }

  ## predict the "best class", i.e., the best solver
  if (parallelize) {
    n.cpus = parallel::detectCores()
    parallelMap::parallelStartMulticore(cpus = n.cpus, level = "mlr.resample", show.info = FALSE)
  }
  set.seed(s + repl - 1L)
  res = resample(learner = lrn, task = task, resampling = resampling.instance, measures = measure.list, models = TRUE)
  if (parallelize) {
    parallelMap::parallelStop()
  }

  # ## compute the resulting performance
  closed.sbs.vbs.gap.model = as.numeric((instance$sbs.cv - res$aggr["runtimes.exp.test.mean"]) / (instance$sbs.cv - instance$vbs))
  ## if the model is performing poorly, we do not have to store the separate models of the fold
  filename = sprintf("%s/intermediate_as_results/final_as_with_repls-ex/classif--%s--%s--%04i--r%02i.RData", instance$wd, job.name, meth, s, repl)
  if (meth == "none") {
    sf = NULL
  }
  save(res, instance, task, lrn, sf, file = filename)

  return(c(vbs = instance$vbs, sbs = instance$sbs, sbs.cv = instance$sbs.cv, setNames(as.list(res$aggr), c("costs.incl.feats", "costs.excl.feats")),
    closed.gap.model = closed.sbs.vbs.gap.model, no.of.feats = length(getTaskFeatureNames(task))))
}

##################################################################################################

## problem design
prob.grid.ecj = expand.grid(
  root.sets = "ecj2018",
  feature.set = c("mst_depth|nng", "mst_depth|nng|autoencoder", "mst_depth|nng|size", "mst_depth|nng|autoencoder|size"),
  tsp.set = "all",
  solver.set = c("eax|lkh", "all"),
  perf.measure = c("par10", "pqr10"),
  normalized.feats= c(TRUE, FALSE),
  learner = c("rpart", "xgboost", "ksvm", "randomForest", "rf"),
  predict.type = c("prob", "response"),
  stringsAsFactors = FALSE
)
attr(prob.grid.ecj, "out.attrs") = NULL

prob.des.ecj = list(
  prob.ecj = prob.grid.ecj
)

##################################################################################################

noFS.grid = expand.grid(
  s = 123L,
  parallelize = c(TRUE, FALSE),
  stringsAsFactors = FALSE
)
attr(noFS.grid, "out.attrs") = NULL

Seq.grid = expand.grid(
  s = 123L,
  maxit = NA_integer_,
  max.features = NA_integer_,
  meth = c("sffs", "sfbs"),
  alpha = 0.0001,
  beta = 0,
  parallelize = c(TRUE, FALSE),
  stringsAsFactors = FALSE
)
attr(Seq.grid, "out.attrs") = NULL

##################################################################################################

## create the algorithm designs
algo.des = list(
  sequential = Seq.grid,
  none = noFS.grid
)

##################################################################################################

## define all experiments

## define the experiments
addProblem(reg = reg, name = "prob.ecj", fun = generateProblem, seed = 123L)
addAlgorithm(reg = reg, name = "none", fun = featselNone)
addAlgorithm(reg = reg, name = "sequential", fun = featselSeq)

## add all experiments with their problem designs and algorithms
addExperiments(reg = reg, prob.designs = prob.des.ecj, algo.designs = algo.des, repls = 25L)

stop()
submitJobs()

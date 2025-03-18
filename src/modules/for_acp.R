# ----------------- Installation de packages ------------------------------
packages <- c("dplyr", "ggplot2", "tidyr", "readr", "explor", "FactoMineR")

# Fonction pour installer les packages manquants
packages_to_install <- packages[!packages %in% installed.packages()[, "Package"]]
if (length(packages_to_install) > 0) {
  install.packages(packages_to_install)
}

# Charger tous les packages
lapply(packages, library, character.only = TRUE)


num_ligne <- function(ind, donnees) match(ind, rownames(donnees), nomatch = 0)
num_col <- function(Var, donnees) match(Var, colnames(donnees), nomatch = 0)

setwd("d:/Projet/formation")

df <- read_csv("data/processed/for_acp.csv")

acp <- PCA(
  df, scale.unit = T, ncp = Inf, 
  quanti.sup = NULL, quali.sup = num_col("SeriousDlqin2yrs", df), graph = F
)
options(shiny.launch.browser = TRUE)
explor(acp)
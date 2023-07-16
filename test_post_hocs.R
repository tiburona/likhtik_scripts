library(lme4)

library(ggplot2)
library(ggeffects)
library(sjPlot)
library(plotly)
library(dplyr)

data_dir <- '/Users/katie/likhtik/data'
graph_dir = file.path('data_dir', 'graphs')

data_file <- file.path(data_dir, 'proportion_score_continuous_trials.csv')
data <- read.csv(data_file, row.names = NULL)
data$animal <- factor(data$animal)
data$unit_num <- factor(data$unit_num)

subset_data = subset(data, category == 'IN' & time_point == 0)

model <- lmer(
  formula = "proportion ~ condition + (1|animal/unit_num)",
  data = subset_data,
)


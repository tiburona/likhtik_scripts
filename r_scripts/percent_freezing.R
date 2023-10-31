library(lme4)
library(dplyr)
library(ggplot2)
library(emmeans)
library(rlang)

freezing_csv <- '/Users/katie/likhtik/data/lfp/percent_freezing/spike_power_mrl.csv'
freezing_data <- read.csv(mrl_power_csv, comment.char="#") 

data_with_freezing <- mrl_freezing_data %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type, animal) %>%
  summarise(
    percent_freezing = mean(percent_freezing, na.rm = TRUE),
    .groups = "drop"
  )

ggplot(data_with_freezing, aes(x = period, y = percent_freezing, color = period_type, shape = period_type)) +
  stat_summary(fun = mean, geom = "point", size = 3, aes(shape = period_type, color = period_type)) +
  stat_summary(fun = mean, geom = "line", aes(group = period_type)) +
  scale_shape_manual(values = c("pretone" = 15, "tone" = 16)) + # 15 is for squares and 16 is for circles
  scale_color_manual(values = c("pretone" = "gray", "tone" = "blue")) +
  labs(x = "Period", y = "Percent Freezing") +
  facet_wrap(~ group, ncol = 1) + # Separate panels for each level of 'group'
  theme_minimal()

freezing.model <- lmer(percent_freezing ~ group * period_type + (1|animal), data = data_with_freezing)
summary(freezing.model)
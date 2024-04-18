
                library(glmmTMB)
                library(lme4)
                library(lmerTest)  # for lmer p-values
                library(readr)
                library(dplyr)
                library(sjPlot)
                library(ggplot2)
                library(emmeans)

                df <- read.csv('/Users/katie/likhtik/IG_INED_Safety_Recall/power/psth_power.csv', comment.char="#")
                subset_df <- subset(df, df$animal %in% c('IG160', 'IG163', 'IG176', 'IG178', 'IG180', 'IG154', 'IG156', 'IG158', 'IG177', 'IG179'))
                #subset_df <- subset(subset_df, subset_df$neuron_quality %in% c('1', '2', '2a', '2b', '2ab'))
                subset_df <- subset(subset_df, subset_df$time_bin < 30)
                
                
                # Convert variables to factors
                factor_vars <- c('unit', 'animal', 'neuron_type', 'group', 'period_type')
                subset_df[factor_vars] <- lapply(subset_df[factor_vars], factor)

                grouped_df <- subset_df %>%
                  group_by(group, neuron_type, period_type, animal, unit, period, event) %>%
                  summarise(rate = mean(rate, na.rm = TRUE), .groups = 'drop')
                
                mixed_model <- lmer(rate ~ group * period_type * neuron_type + (1|animal/unit/period), data = grouped_df)
                
                psth_plot = emmip(mixed_model, group ~ period_type | neuron_type, CIs = FALSE) + 
                  labs(y = "Predicted Rate") +
                  scale_color_manual(values = c("control" = "green", "defeat" = "orange"))
                

               
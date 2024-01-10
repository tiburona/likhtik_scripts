
library(ggplot2)
library(dplyr)
library(nlme)




csv_dir = '/Users/katie/likhtik/IG_INED_Safety_Recall/power'
csv_name = 'PFC_Theta_Power.csv'
csv_file = paste(csv_dir, csv_name, sep='/')
df <- read.csv(csv_file, comment.char="#") 

factor_vars <- c('animal', 'group', 'block_type')
df[factor_vars] <- lapply(df[factor_vars], factor)

data_early_periods <- df %>%
  filter(block < 2)

summarized_df <- data_early_periods %>%
  group_by(group, block_type, block) %>%
  summarize(avg_pl_theta_1_power = mean(pl_theta_1_power))

ggplot(summarized_df, aes(x = block, y = avg_pl_theta_1_power, group = block_type, color = block_type)) +
  geom_line() +
  geom_point(aes(shape = block_type)) +
  scale_color_manual(values = c("tone" = "blue", "pretone" = "gray")) +
  scale_shape_manual(values = c("tone" = 16, "pretone" = 15)) + # 16 is circle, 15 is square
  facet_wrap(~ group) +
  labs(x = "Block", y = "Average PL Theta 1 Power") +
  theme_minimal()
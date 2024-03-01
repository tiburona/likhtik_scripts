library(lme4)
library(nlme)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(emmeans)


power_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/power', 'power_pre_pip.csv', sep='/')
power_data <- read.csv(power_csv, comment.char="#") 

power_data$animal = factor(power_data$animal)
power_data$block_type = factor(power_data$block_type)
power_data$group = factor(power_data$group)

power_data <- power_data %>%
  group_by(block, group, block_type, animal) %>%
  summarise(
    hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE),
    bla_theta_1_power = mean(bla_theta_1_power, na.rm = TRUE),
    pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )



data_early_blocks <- power_data %>%
  filter(block < 2)



### Models ####


bla_model <- lmer(bla_theta_1_power ~ group * block_type + (1|animal/block), data=power_data)
summary(bla_model)
bla_plot = emmip(bla_model, group ~ block_type, CIs = FALSE) + 
  labs(y = "Predicted BLA Power") +
  scale_color_manual(values = c("control" = "green", "defeat" = "orange"))

bla_early_block_model <- lmer(bla_theta_1_power ~ group * block_type + (1|animal/block), data=data_early_blocks)
summary(bla_early_block_model)
bla_early_plot = emmip(bla_early_block_model, group ~ block_type, CIs = FALSE) + 
  labs(y = "Predicted BLA Power (first two blocks)") +
  scale_color_manual(values = c("control" = "green", "defeat" = "orange"))

bla_lm_model <- lm(bla_theta_1_power ~ group * block_type * block, data=data_early_blocks)
summary(bla_lm_model)


pl_model <- lmer(pl_theta_1_power ~ group * block_type + (1|animal/block), data=power_data)
summary(pl_model)
pl_plot = emmip(pl_model, group ~ block_type, CIs = FALSE) + 
  labs(y = "Predicted PL Power") +
  scale_color_manual(values = c("control" = "green", "defeat" = "orange"))

pl_early_block_model <- lmer(pl_theta_1_power ~ group * block_type + (1|animal/block), data=data_early_blocks)
summary(pl_early_block_model)
pl_early_plot = emmip(pl_early_block_model, group ~ block_type, CIs = FALSE) + 
  labs(y = "Predicted PL Power (first two blocks)") +
  scale_color_manual(values = c("control" = "green", "defeat" = "orange"))

pl_lm_model <- lm(pl_theta_1_power ~ group * block_type * block, data=data_early_blocks)
summary(pl_lm_model)


hpc_model <- lmer(hpc_theta_1_power ~ group * block_type + (1|animal/block), data=power_data)
summary(hpc_model)
hpc_plot = emmip(hpc_model, group ~ block_type, CIs = FALSE) + 
  labs(y = "Predicted HPC Power") +
  scale_color_manual(values = c("control" = "green", "defeat" = "orange"))

hpc_early_block_model <- lmer(hpc_theta_1_power ~ group * block_type + (1|animal/block), data=data_early_blocks)
summary(hpc_early_block_model)
hpc_early_plot = emmip(hpc_early_block_model, group ~ block_type, CIs = FALSE) + 
  labs(y = "Predicted HPC Power (first two blocks)") +
  scale_color_manual(values = c("control" = "green", "defeat" = "orange"))

hpc_lm_model <- lm(hpc_theta_1_power ~ group * block_type * block, data=data_early_blocks)
summary(hpc_lm_model)


### Power relationships to each other ###

mean_bla <- mean(mean_data$mean_hpc_theta_1_power, na.rm = TRUE)
sd_bla <- sd(mean_data$mean_hpc_theta_1_power, na.rm = TRUE)
mean_hpc <- mean(mean_data$mean_hpc_theta_1_power, na.rm = TRUE)
sd_hpc <- sd(mean_data$mean_hpc_theta_1_power, na.rm = TRUE)


bla_pl_model <- lmer(mean_pl_theta_1_power ~ group * block_type * mean_bla_theta_1_power + (1|animal), data=mean_data)
summary(bla_pl_model)

bla_pl_plot <- emmip(bla_pl_model, block_type ~ mean_bla_theta_1_power | group,
                     at = list(mean_bla_theta_1_power = c(mean_bla - sd_bla, mean_bla, mean_bla + sd_bla)),
                     CIs = FALSE) + 
  labs(y = "Predicted PL Power") 


hpc_pl_model <- lmer(mean_pl_theta_1_power ~ group * block_type * mean_hpc_theta_1_power + (1|animal), data=mean_data)
summary(hpc_pl_model)



hpc_pl_plot <- emmip(hpc_pl_model, block_type ~ mean_hpc_theta_1_power | group,
                     at = list(mean_hpc_theta_1_power = c(mean_hpc - sd_hpc, mean_hpc, mean_hpc + sd_hpc)),
                     CIs = FALSE) + 
  labs(y = "Predicted PL Power") 









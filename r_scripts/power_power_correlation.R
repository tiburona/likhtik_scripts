library(dplyr)


max_correlation_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/correlation', 'lag_of_max_correlation.csv', sep='/')
max_corr_data <- read.csv(max_correlation_csv, comment.char="#") 


bla_pl_control_tone = subset(max_corr_data, max_corr_data$group == 'control' & max_corr_data$period_type == 'tone' & !is.na(max_corr_data$bla_pl_theta_1_correlation))
bla_pl_control_pretone = subset(max_corr_data, max_corr_data$group == 'control' & max_corr_data$period_type == 'pretone' & !is.na(max_corr_data$bla_pl_theta_1_correlation))
bla_pl_defeat_tone = subset(max_corr_data, max_corr_data$group == 'defeat' & max_corr_data$period_type == 'tone' & !is.na(max_corr_data$bla_pl_theta_1_correlation))
bla_pl_defeat_pretone = subset(max_corr_data, max_corr_data$group == 'defeat' & max_corr_data$period_type == 'pretone' & !is.na(max_corr_data$bla_pl_theta_1_correlation))


wilcox.test(bla_pl_control_pretone$bla_pl_theta_1_correlation, mu=0, alternative="two.sided")
wilcox.test(bla_pl_control_tone$bla_pl_theta_1_correlation, mu=0, alternative="two.sided")
wilcox.test(bla_pl_defeat_pretone$bla_pl_theta_1_correlation, mu=0, alternative="two.sided")
wilcox.test(bla_pl_defeat_tone$bla_pl_theta_1_correlation, mu=0, alternative="two.sided")
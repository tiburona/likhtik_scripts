library(lme4)
library(dplyr)
library(lmerTest)


csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/coherence', 'coherence.csv', sep='/')
data <- read.csv(csv, comment.char="#") 


data$bla_pl_3_6_coherence <- data$bla_pl__3_.6__coherence
data$bla_pl_6_12_coherence <- data$bla_pl__6_.12__coherence
data$bla_hpc_3_5_coherence <- data$bla_hpc__3_.5__coherence
data$bla_hpc_5_12_coherence <- data$bla_hpc__5_.12__coherence
data$hpc_pl_3_5_coherence <- data$hpc_pl__3_.5__coherence
data$hpc_pl_5_12_coherence <- data$hpc_pl__5_.12__coherence



### BLA_PL 3-6

clean_data <- data[!is.na(data$bla_pl_3_6_coherence), ]
model <- lmer(bla_pl_3_6_coherence ~ group*period_type + (1|animal), data = clean_data)
summary(model)

plot(residuals(model) ~ fitted(model))
abline(h = 0, col = "red")


plot = emmip(model, group ~ period_type, CIs = FALSE) + 
  labs(y = "Predicted Coherence") +
  scale_color_manual(values = c("control" = "purple", "defeat" = "orange"))

print(plot)

### BLA_PL 6-12

clean_data <- data[!is.na(data$bla_pl_6_12_coherence), ]
model <- lmer(bla_pl_6_12_coherence ~ group*period_type + (1|animal), data = clean_data)
summary(model)

plot(residuals(model) ~ fitted(model))
abline(h = 0, col = "red")

plot = emmip(model, group ~ period_type, CIs = FALSE) + 
  labs(y = "Predicted Coherence") +
  scale_color_manual(values = c("control" = "purple", "defeat" = "orange"))

print(plot)


### BLA_HPC 3-5

clean_data <- data[!is.na(data$bla_hpc_3_5_coherence), ]
model <- lmer(bla_hpc_3_5_coherence ~ group*period_type + (1|animal), data = clean_data)
summary(model)

plot(residuals(model) ~ fitted(model))
abline(h = 0, col = "red")

plot = emmip(model, group ~ period_type, CIs = FALSE) + 
  labs(y = "Predicted Coherence") +
  scale_color_manual(values = c("control" = "purple", "defeat" = "orange"))

### BLA_HPC 5-12

clean_data <- data[!is.na(data$bla_hpc_5_12_coherence), ]
model <- lmer(bla_hpc_5_12_coherence ~ group*period_type + (1|animal), data = clean_data)
summary(model)

plot(residuals(model) ~ fitted(model))
abline(h = 0, col = "red")

plot = emmip(model, group ~ period_type, CIs = FALSE) + 
  labs(y = "Predicted Coherence") +
  scale_color_manual(values = c("control" = "purple", "defeat" = "orange"))


### HPC_PL 3-5

clean_data <- data[!is.na(data$hpc_pl_3_5_coherence), ]
model <- lmer(hpc_pl_3_5_coherence ~ group*period_type + (1|animal), data = clean_data)
summary(model)

plot(residuals(model) ~ fitted(model))
abline(h = 0, col = "red")

plot = emmip(model, group ~ period_type, CIs = FALSE) + 
  labs(y = "Predicted Coherence") +
  scale_color_manual(values = c("control" = "purple", "defeat" = "orange"))

### HPC_PL 5-12

clean_data <- data[!is.na(data$hpc_pl_5_12_coherence), ]
model <- lmer(hpc_pl_5_12_coherence ~ group*period_type + (1|animal), data = clean_data)
summary(model)

plot(residuals(model) ~ fitted(model))
abline(h = 0, col = "red")

plot = emmip(model, group ~ period_type, CIs = FALSE) + 
  labs(y = "Predicted Coherence") +
  scale_color_manual(values = c("control" = "purple", "defeat" = "orange"))




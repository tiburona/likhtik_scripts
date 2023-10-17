csv_dir = '/Users/katie/likhtik/data/lfp/power'
csv_name = 'power_bla_delta_power_bla_theta_1_power_bla_theta_2_lfp_power_bla_by_period.csv'

csv_file = paste(csv_dir, csv_name, sep='/')
data <- read.csv(csv_file, comment.char="#") 
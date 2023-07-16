cd('/Users/katie/likhtik')

python_random_nums = load('./python_random_nums.txt');
python_random_autocorr = load('./python_random_autocorr.txt');

python_expt_nums = load('./python_expt_numbers.txt');
python_expt_autocorr = load('./python_expt_autocorr.txt');

matlab_random_autocorr = xcorr(python_random_nums, 99, 'coeff');
matlab_expt_autocorr = xcorr(python_expt_nums, 99, 'coeff');

demeaned_nums = python_expt_nums - mean(python_expt_nums);
matlab_demeaned_autocorr = xcorr(demeaned_nums, 99, 'coeff');

plot1 = figure(1);
plot(python_random_autocorr(2:end))
title('Python Random Autocorrelation')

plot2 = figure(2);
plot(matlab_random_autocorr(101:end))
title('Matlab Random Autocorrelation')

plot3 = figure(3);
plot(python_expt_autocorr(2:end))
title('Python Experiment Autocorrelation')

plot4 = figure(4);
plot(matlab_expt_autocorr(101:end))
title('Matlab Experiment Autocorrelation')

plot5 = figure(5);
plot(matlab_demeaned_autocorr(101:end))
title('Matlab Demeaned Autocorrelation')


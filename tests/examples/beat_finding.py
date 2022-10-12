from thebeat.stats import acf_plot, acf_df, get_ugof, acf_values
from thebeat.linguistic import generate_trimoraic_sequence
import scipy.signal


# generate example sequence
seq = generate_trimoraic_sequence(10)
# add some Gaussian noise
seq.add_noise_gaussian(25)

# check out ACF plot and get ACF dataframe
acf_plot(seq, smoothe_width=100, smoothe_sd=30)
print(scipy.signal.find_peaks(acf_values(seq, 1, 100, 30)))
print(acf_df(seq))

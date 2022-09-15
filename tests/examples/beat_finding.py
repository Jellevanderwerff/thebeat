from combio.stats import acf_plot, acf_df, get_ugof
from combio.linguistic import generate_trimoraic_sequence

# generate example sequence
seq = generate_trimoraic_sequence(10)

# check out ACF plot and get ACF dataframe
acf_plot(seq)
df = acf_df(seq)
print(df)
# calculate ugof for suspected IOI
ugof = get_ugof(seq, 750)
print(f"UGOF: {ugof}")

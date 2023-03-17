import cudf

# create a cudf DataFrame
df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

# select a subset of the data
subset_df = df[["a", "c"]]

# filter the data based on a condition
filtered_df = df[df["b"] > 4]

# perform a groupby operation and calculate the mean for each group
grouped_df = df.groupby("a").mean()

# join two dataframes based on a common column
other_df = cudf.DataFrame({"a": [1, 2, 3], "d": [10, 11, 12]})
joined_df = df.merge(other_df, on="a")

# print the resulting dataframes
print(df)
print(subset_df)
print(filtered_df)
print(grouped_df)
print(joined_df)

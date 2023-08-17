standardize = lambda x: (x - x.mean()) / x.std()
unstandardize = lambda x, mean, std: x * std + mean

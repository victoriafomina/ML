import pandas


def get_normalize_data(file_csv: str):
    """Get normalize data."""
    data = pandas.read_csv(file_csv)
    data = remove_string_data(data)

    return (data - data.min()) / (data.max() - data.min())


def remove_string_data(data):
    """Removing and replace to number string features."""
    data.drop(
        [
            "car_ID",
            "CarName",
            "symboling",
            "fueltype",
            "aspiration",
            "enginelocation",
            "enginetype",
            "carbody",
            "drivewheel",
            "doornumber",
            "carheight",
            "stroke",
            "peakrpm",
            "compressionratio",
            "carlength",
            "carwidth",
            "wheelbase",
            "curbweight",
            "highwaympg",
            "cylindernumber",
            "fuelsystem",
        ],
        axis=1,
        inplace=True,
    )

    return data

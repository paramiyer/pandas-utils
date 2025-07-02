def cleaning_pipeline(df, missing_threshold=0.4, zscore_threshold=3):
    """
    Cleans a DataFrame by:
    1. Dropping columns with missing values > missing_threshold
    2. Imputing remaining missing values with median
    3. Removing rows with Z-score outliers (absolute Z > zscore_threshold)

    Parameters:
        df (pd.DataFrame): The input DataFrame
        missing_threshold (float): Max allowed missing % per column (0.0 - 1.0)
        zscore_threshold (float): Threshold for Z-score based outlier removal

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df = df.copy()

    # Step 1: Drop columns with too many missing values
    df = df.loc[:, df.isnull().mean() <= missing_threshold]

    # Step 2: Impute remaining missing values using median
    df = df.fillna(df.median(numeric_only=True))

    # Step 3: Remove outliers using Z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(zscore(df[numeric_cols]))

    # Keep rows where all z-scores are below threshold
    df = df[(z_scores < zscore_threshold).all(axis=1)]

    return df

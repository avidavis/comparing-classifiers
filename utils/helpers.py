import pandas as pd
import numpy as np


def profile_features(df):
    """
    Profiles every column in a DataFrame. Returns a summary DataFrame 
    with dtype, inferred type, unique count, null count, and sample values.
    Pure facts, no opinions.
    
    Arguments:
    - df: pandas DataFrame to profile.
    
    Returns:
    - DataFrame with one row per column.
    """
    # We'll collect a dict for each column, then build a DataFrame at the end
    records = []
    
    # Loop through every column in the DataFrame
    for col in df.columns:
        # Grab the column as a Series so we can inspect it
        series = df[col]
        
        # Count how many distinct non-null values exist in this column
        n_unique = series.nunique()
        
        # Count how many null/NaN values exist
        n_null = series.isnull().sum()
        
        # Express nulls as a percentage of total rows for quick triage
        pct_null = round(n_null / len(df) * 100, 2)
        
        # Store the pandas dtype as a string (e.g. "int64", "object", "float64")
        dtype = str(series.dtype)
        
        # Grab up to 5 unique non-null values so we can eyeball what's in the column
        # .unique() returns a numpy array, we slice first 5 and convert to list
        sample_vals = series.dropna().unique()[:5].tolist()
        
        # Use our helper function to figure out the semantic type
        # (numeric, categorical, binary, etc.) based on dtype and unique count
        inferred = _infer_type(series, n_unique)
        
        # Pack everything into a dict and add to our list
        records.append({
            "column": col,
            "dtype": dtype,
            "inferred_type": inferred,
            "n_unique": n_unique,
            "n_null": n_null,
            "pct_null": pct_null,
            "sample_values": sample_vals
        })
    
    # Convert list of dicts into a DataFrame, using column name as the index
    return pd.DataFrame(records).set_index("column")


def _infer_type(series, n_unique):
    """
    Infers the semantic type of a pandas Series. This goes beyond the raw 
    dtype to figure out what the data actually represents.
    
    Arguments:
    - series: a single pandas Series (one column).
    - n_unique: pre-computed count of unique non-null values.
    
    Returns one of:
    - "numeric": continuous numbers that should be scaled
    - "binary": exactly 2 possible values (yes/no, 0/1, etc.)
    - "categorical": discrete labels with <= 20 unique values
    - "high_cardinality_categorical": discrete labels with > 20 values
    - "low_cardinality_numeric": integers with very few unique values
      (might be ordinal or might need to be treated as categorical)
    - "datetime": date or timestamp column
    - "unknown": fallback for anything we can't classify
    """
    # Get the pandas dtype for this series
    dtype = series.dtype
    
    # Check if pandas already recognizes this as a datetime column
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    
    # Check for boolean columns (True/False)
    if pd.api.types.is_bool_dtype(dtype):
        return "binary"
    
    # Handle numeric columns (int64, float64, etc.)
    if pd.api.types.is_numeric_dtype(dtype):
        # Even though it's numeric, if there are only 2 unique values
        # it's effectively binary (e.g. 0 and 1)
        unique_vals = set(series.dropna().unique())
        if len(unique_vals) == 2:
            return "binary"
        
        # If it's an integer with 10 or fewer unique values, it might be
        # categorical disguised as numbers (e.g. education level 1-4)
        # Flag it so the user can decide how to treat it
        if pd.api.types.is_integer_dtype(dtype) and n_unique <= 10:
            return "low_cardinality_numeric"
        
        # Otherwise it's a standard numeric feature
        return "numeric"
    
    # Handle object/string columns (pandas stores text as "object" dtype)
    if dtype == "object" or pd.api.types.is_string_dtype(dtype):
        # Only 2 unique string values means binary (e.g. "yes"/"no")
        if n_unique == 2:
            return "binary"
        
        # More than 20 unique strings is high cardinality
        # One-hot encoding would create too many columns
        if n_unique > 20:
            return "high_cardinality_categorical"
        
        # Standard categorical with a manageable number of classes
        return "categorical"
    
    # If nothing matched, flag it for manual review
    return "unknown"


def get_feature_groups(df, target_col=None):
    """
    Groups columns by their inferred type. Returns a dict of column name lists.
    This output plugs directly into ColumnTransformer setup.
    
    Arguments:
    - df: pandas DataFrame.
    - target_col: name of the target column to exclude from groups (optional).
    
    Returns:
    - dict with keys like "numeric", "categorical", "binary", etc.
      Each value is a list of column names belonging to that group.
    """
    # Run profile_features first to get the inferred types
    profile = profile_features(df)
    
    # Remove the target column from the profile so it doesn't end up
    # in any feature group (it's what we're predicting, not an input)
    if target_col and target_col in profile.index:
        profile = profile.drop(target_col)
    
    # Initialize empty lists for each possible group
    groups = {
        "numeric": [],
        "categorical": [],
        "binary": [],
        "high_cardinality": [],
        "low_cardinality_numeric": [],
        "other": []
    }
    
    # Map each inferred_type string to the corresponding group key
    # This lets us handle slight naming differences in one place
    type_map = {
        "numeric": "numeric",
        "categorical": "categorical",
        "binary": "binary",
        "high_cardinality_categorical": "high_cardinality",
        "low_cardinality_numeric": "low_cardinality_numeric",
        "datetime": "other",
        "boolean": "binary",
        "unknown": "other"
    }
    
    # Sort each column into the right group based on its inferred type
    for col, row in profile.iterrows():
        # Look up which bucket this type maps to, default to "other"
        bucket = type_map.get(row["inferred_type"], "other")
        groups[bucket].append(col)
    
    # Only return groups that actually have columns in them
    # Keeps the output clean and easy to iterate over
    return {k: v for k, v in groups.items() if v}


def suggest_preprocessing(profile_df):
    """
    Takes a profile DataFrame (from profile_features) and adds a 
    'suggestion' column with preprocessing recommendations.
    
    Arguments:
    - profile_df: output of profile_features().
    
    Returns:
    - Same DataFrame with an added 'suggestion' column.
    """
    # We'll build a list of suggestion strings, one per column
    suggestions = []
    
    # Walk through each column's profile row
    for col, row in profile_df.iterrows():
        inferred = row["inferred_type"]
        n_unique = row["n_unique"]
        pct_null = row["pct_null"]
        
        # Collect multiple suggestion parts, then join them at the end
        parts = []
        
        # --- Null handling suggestions ---
        # If more than half the column is null, it might not be worth keeping
        if pct_null > 50:
            parts.append(f"consider dropping ({pct_null}% null)")
        # If some nulls exist but not too many, imputation is the way to go
        elif pct_null > 0:
            parts.append(f"impute ({pct_null}% null)")
        
        # --- Type-based preprocessing suggestions ---
        if inferred == "numeric":
            # Continuous numbers need scaling for distance-based models (KNN, SVM)
            parts.append("scale (StandardScaler)")
            
        elif inferred == "binary":
            # Two-value columns just need 0/1 encoding
            parts.append("label encode (0/1)")
            
        elif inferred == "categorical":
            # Discrete labels with manageable cardinality get one-hot encoded
            parts.append(f"one-hot encode ({n_unique} classes)")
            
        elif inferred == "high_cardinality_categorical":
            # Too many unique values for one-hot, warn the user
            parts.append(f"review: {n_unique} unique values, one-hot may explode dimensionality")
            
        elif inferred == "low_cardinality_numeric":
            # Could go either way, user needs to decide based on domain knowledge
            parts.append(f"check if ordinal or categorical ({n_unique} unique values)")
            
        elif inferred == "datetime":
            # Raw dates aren't useful to models, need to extract components
            parts.append("extract features (month, day, year) or drop")
        
        # --- Sentinel value detection ---
        # Some datasets use -1 to mean "not applicable" (like pdays in the bank data)
        # This would throw off scaling if treated as a real number
        if inferred in ("numeric", "low_cardinality_numeric"):
            sample = row["sample_values"]
            if isinstance(sample, list) and -1 in sample:
                parts.append("investigate -1 sentinel value")
        
        # Join all parts with a pipe separator, or note no action needed
        suggestions.append(" | ".join(parts) if parts else "no action needed")
    
    # Copy the original profile so we don't modify it in place
    result = profile_df.copy()
    
    # Add the suggestions as a new column
    result["suggestion"] = suggestions
    
    return result


def summarize_target(df, target_col):
    """
    Summarizes the target variable. Handles both categorical 
    and numeric targets. Checks for class imbalance and recommends 
    appropriate evaluation metrics.
    
    Arguments:
    - df: pandas DataFrame.
    - target_col: name of the target column.
    """
    # Pull the target column out as a Series
    y = df[target_col]
    
    # Count distinct values to determine if this is classification or regression
    n_unique = y.nunique()
    
    # Print header info
    print(f"Target: '{target_col}'")
    print(f"Type: {y.dtype} | Unique values: {n_unique}")
    print("-" * 40)
    
    # Get raw counts for each class (e.g. "yes": 5289, "no": 39922)
    counts = y.value_counts()
    
    # Get proportions as percentages (e.g. "yes": 11.7%, "no": 88.3%)
    props = y.value_counts(normalize=True) * 100
    
    # Print each class with its count and percentage
    for cls in counts.index:
        print(f"  {cls}: {counts[cls]} samples ({props[cls]:.1f}%)")
    
    # Find the percentage of the largest class for imbalance check
    majority_pct = props.max()
    
    print("-" * 40)
    
    # If there are many unique numeric values, this is probably regression
    # not classification, so give different summary stats
    if n_unique > 10 and pd.api.types.is_numeric_dtype(y):
        print("This looks like a regression target, not classification.")
        print(f"  Mean: {y.mean():.2f} | Median: {y.median():.2f} | Std: {y.std():.2f}")
        return
    
    # --- Class imbalance assessment ---
    # 80%+ majority class means accuracy is basically useless as a metric
    # A model that always predicts the majority class gets 80%+ accuracy
    if majority_pct > 80:
        print(f"HIGHLY IMBALANCED ({majority_pct:.1f}% majority class)")
        print("  Accuracy will be misleading.")
        print("  Use F1, Precision, Recall, or AUC-ROC.")
        print("  Consider SMOTE or class_weight='balanced'.")
    
    # 60-80% is a gray zone, worth noting but not critical
    elif majority_pct > 60:
        print(f"SLIGHTLY IMBALANCED ({majority_pct:.1f}% majority class)")
        print("  Watch the confusion matrix closely.")
        print("  F1-score is a safer metric than accuracy.")
    
    # Under 60% split means classes are roughly even
    else:
        print("WELL BALANCED")
        print("  Accuracy is a reasonable metric here.")


def data_overview(df, target_col=None):
    """
    One-call overview of a DataFrame. Profiles all features, 
    groups them by type, suggests preprocessing, and summarizes 
    the target if provided.
    
    Arguments:
    - df: pandas DataFrame.
    - target_col: name of the target column (optional).
    
    Returns:
    - tuple of (profile_df, feature_groups, suggestions_df)
    """
    # --- Section 1: Feature Profile ---
    # Show the raw facts about every column
    print("=" * 60)
    print("  FEATURE PROFILE")
    print("=" * 60)
    profile = profile_features(df)
    
    # Run suggestions on top of the profile to add the recommendation column
    suggestions = suggest_preprocessing(profile)
    
    # Print the full table so everything is visible in notebook output
    print(suggestions.to_string())
    
    # --- Section 2: Feature Groups ---
    # Group columns by type for easy ColumnTransformer setup
    print(f"\n{'=' * 60}")
    print("  FEATURE GROUPS")
    print("=" * 60)
    groups = get_feature_groups(df, target_col)
    
    # Print each group and its column names
    for group_name, cols in groups.items():
        print(f"\n  {group_name}: {cols}")
    
    # --- Section 3: Target Summary ---
    # Only run if a target column was specified
    if target_col:
        print(f"\n{'=' * 60}")
        print("  TARGET SUMMARY")
        print("=" * 60)
        summarize_target(df, target_col)
    
    # Return all three outputs so the user can work with them downstream
    # e.g. groups["numeric"] feeds right into ColumnTransformer
    return profile, groups, suggestions

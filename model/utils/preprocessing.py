import pandas as pd

def preprocess_input_data(data):
    """
    Preprocesses the incoming data to match the feature set used during model training.
    """
    # Convert the input data into a DataFrame (assuming 'data' is a dictionary or a JSON object)
    df = pd.DataFrame([data])
    
    # Define a mapping for obsession and compulsion categories
    obsession_mapping = {
        'Contamination': 'Obsession Type_Contamination',
        'Harm-related': 'Obsession Type_Harm-related',
        'Hoarding': 'Obsession Type_Hoarding',
        'None': 'Obsession Type_None',
        'Religious': 'Obsession Type_Religious',
        'Symmetry': 'Obsession Type_Symmetry'
    }

    compulsion_mapping = {
        'None': 'Compulsion Type_None',
        'Checking': 'Compulsion Type_Checking',
        'Counting': 'Compulsion Type_Counting',
        'Ordering': 'Compulsion Type_Ordering',
        'Praying': 'Compulsion Type_Praying',
        'Washing': 'Compulsion Type_Washing'
    }

    # Map obsession and compulsion types to the new columns
    df['Obsession Type'] = obsession_mapping.get(data['Obsession Type'], 'Obsession Type_None')
    df['Compulsion Type'] = compulsion_mapping.get(data['Compulsion Type'], 'Compulsion Type_None')

    # One-hot encode categorical features (adjust column names based on actual categories)
    df = pd.get_dummies(df, columns=['Obsession Type', 'Compulsion Type'], drop_first=True)
    
    # List of features that the model expects, based on training data
    required_columns = [
        'Age', 'Duration of Symptoms (months)', 'Depression Diagnosis', 'Anxiety Diagnosis', 
        'Obsession Type_Contamination', 'Obsession Type_Harm-related', 'Obsession Type_Hoarding',
        'Obsession Type_None', 'Obsession Type_Religious', 'Obsession Type_Symmetry',
        'Compulsion Type_Checking', 'Compulsion Type_Counting', 'Compulsion Type_None', 
        'Compulsion Type_Ordering', 'Compulsion Type_Praying', 'Compulsion Type_Washing'
    ]
    
    # Ensure the input data contains exactly the columns that the model expects
    # Reindex the dataframe to match required_columns, filling missing columns with 0
    df = df.reindex(columns=required_columns, fill_value=0)
    
    # Ensure that the columns align even if extra columns were introduced
    missing_columns = [col for col in required_columns if col not in df.columns]
    for col in missing_columns:
        df[col] = 0  # Add missing columns with zero values
    
    print("Preprocessed data:", df)  # Debug: Log the preprocessed data
    
    return df

import os
import pandas as pd

# Directory containing the prediction CSV files
prediction_dir = "Prediction"

# List to store results for each file
results = []

# Read each CSV file in the prediction directory
for filename in os.listdir(prediction_dir):
    if filename.endswith(".csv"):
        # Read the CSV file
        filepath = os.path.join(prediction_dir, filename)
        df = pd.read_csv(filepath)
        
        # Count the occurrences of each tag
        tag_counts = df['prediction'].value_counts().to_dict()
        
        # Calculate the average of the tags
        avg = df['prediction'].mean()
        
        # Append the results to the list
        results.append({
            'file name': filename.replace('.csv', ''),
            '-1': tag_counts.get(2, 0),  # Map '2' to '-1'
            '0': tag_counts.get(0, 0),
            '1': tag_counts.get(1, 0),
            'avg': (tag_counts.get(1, 0) - tag_counts.get(2, 0))/df['prediction'].count()
        })

# Create a DataFrame from the results
result_df = pd.DataFrame(results)

# Save the results to Result.csv
result_df.to_csv("Result.csv", index=False)

print("Results have been saved to Result.csv")

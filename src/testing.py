import pandas as pd

# Example data
data = [
    [1.0, 0.0, -0.020833, 0.020833, 0.0, 2.875],
    [0.0, 0.0, -10.3125, 0.3125, 1.0, 23.125],
    [0.0, 1.0, 0.208333, -0.008333, 0.0, 1.25],
    [0.0, 0.0, 1.708333, 0.091667, 0.0, 34.25]
]

# Define column and row labels
column_labels = ['x0', 'x1', 'x2', 'x3', 'x4', 'b']
row_labels = ['x0', 'z2', 'x1', 'OBJ']

# Create the DataFrame
tableau = pd.DataFrame(data, columns=column_labels, index=row_labels)

# Display the DataFrame
print(tableau)

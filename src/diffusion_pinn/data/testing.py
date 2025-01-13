import numpy as np

def read_csv_numpy():
    filename = "intensity_time_series_spatial_temporal.csv"
    try:
        # Load data using numpy directly
        data = np.genfromtxt(filename, delimiter=',', skip_header=1,
                            dtype=float)
        print("Data shape:", data.shape)
        print("First few rows:")
        print(data[:5])
        
        # Extract columns
        x = data[:, 0]  # first column
        y = data[:, 1]  # second column
        t = data[:, 2]  # third column
        intensity = data[:, 3]  # fourth column
        
        print("\nUnique time steps:", np.unique(t))
        print("X range:", np.min(x), "to", np.max(x))
        print("Y range:", np.min(y), "to", np.max(y))
        print("Intensity range:", np.min(intensity), "to", np.max(intensity))
        
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    read_csv_numpy()


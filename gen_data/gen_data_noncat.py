import numpy as np
import pandas as pd

def generate_non_categorical_data(n_samples=5000):
    """
    Generate simulated continuous data for time steps t = 1, 2, ..., 8.
    
    For each time step, we create 7 attributes (X1 through X7) that mimic the dependency
    patterns from the original Bernoulli (categorical) formulation but using continuous distributions.
    
    Returns:
        data: A dictionary with keys 't', 'X1', ..., 'X7' each mapping to a list of generated values.
    """
    # Define time steps from 1 to 8.
    # This will allow us to simulate data evolving over time.
    time_steps = np.arange(1, 9)
    
    # Initialize a dictionary to store our generated data.
    # The key "t" will record the time step for each sample, while X1-X7 will store attribute values.
    data = {
        "t": [],
        "X1": [],
        "X2": [],
        "X3": [],
        "X4": [],
        "X5": [],
        "X6": [],
        "X7": []
    }
    
    # Loop over each time step to generate data corresponding to that time point.
    for t in time_steps:
        # --- X1: Baseline Attribute ---
        # In the Bernoulli version, X1 ~ Bernoulli(0.2). For continuous data, we use a normal distribution.
        # This provides a baseline signal with natural variability (mean=0, std=1).
        X1 = np.random.normal(loc=0, scale=1, size=n_samples)
        
        '''# --- X2: Independent Attribute ---
        # Originally, X2 ~ Bernoulli(0.1) and is independent of X1.
        # Here, we generate X2 from a uniform distribution between -1 and 1 to ensure no dependency on X1.
        X2 = np.random.uniform(low=-1, high=1, size=n_samples)'''

        # --- X2: Independent Attribute ---
        #Switching to normal distribution to ensure that covariance is a predictable measure of independence. 
        X2 = np.random.normal(loc=0, scale=1, size=n_samples)

        
        # --- X3: Highly Dependent on X1 ---
        # The Bernoulli formulation is X3 ~ Bernoulli(0.9) * X1, making X3 highly dependent on X1.
        # For continuous data, we mimic this by setting X3 as a strong linear function of X1 (multiplied by 2)
        # and adding small noise to simulate minor variability.
        noise3 = np.random.normal(loc=0, scale=0.5, size=n_samples)  # Low noise for high dependency.
        X3 = 2 * X1 + noise3
        
        # --- X4: Increasing Dependence on X1 Over Time ---
        # Categorical version: X4 ~ Bernoulli(0.1 + 0.1*t) * X1, meaning the dependency on X1 grows over time.
        # Here, we let the coefficient multiplying X1 increase with time (t) to simulate that behavior.
        slope4 = 0.5 + 0.5 * t  # As t increases, the influence of X1 on X4 becomes stronger.
        noise4 = np.random.normal(loc=0, scale=1.0, size=n_samples)  # Moderate noise to allow variability.
        X4 = slope4 * X1 + noise4
        
        # --- X5: Decreasing Dependence on X1 Over Time ---
        # In the Bernoulli model, X5 ~ Bernoulli(0.9 - 0.1*t) * X1 shows a diminishing dependency on X1 as time increases.
        # We simulate this by letting the coefficient decrease with t.
        slope5 = 4.5 - 0.5 * t  # The coefficient reduces as t increases.
        noise5 = np.random.normal(loc=0, scale=1.0, size=n_samples)
        X5 = slope5 * X1 + noise5
        
        # --- X6: Shifting Dependence on X1 ---
        # Categorical description: X6 ~ Bernoulli(0.1 + 0.2|t-4|) * X1, meaning dependency shifts based on distance from t = 4.
        # We mirror this by setting the coefficient based on |t - 4|.
        coef6 = 1 + 0.3 * abs(t - 4)  # The dependency adjusts as we move away from t=4.
        noise6 = np.random.normal(loc=0, scale=1.0, size=n_samples)
        X6 = coef6 * X1 + noise6
        
        # --- X7: Mixed Behavior Based on Time ---
        # In the Bernoulli setup:
        #   - For t < 8: X7 ~ Bernoulli(0.1) (independent of X1).
        #   - For t = 8: X7 ~ Bernoulli(0.9) * X1 (suddenly becomes highly dependent on X1).
        # In our continuous analog, we simulate independent behavior for t < 8 and dependency for t = 8.
        if t < 8:
            # For t < 8, generate X7 as an independent normal variable.
            X7 = np.random.normal(loc=0, scale=1.0, size=n_samples)
        else:
            # For t = 8, make X7 highly dependent on X1 by multiplying X1 by a factor and adding low noise.
            noise7 = np.random.normal(loc=0, scale=0.5, size=n_samples)
            X7 = 3 * X1 + noise7
        
        # Append the generated samples to our data dictionary.
        # We also append the corresponding time step using np.repeat.
        data["t"] += list(np.repeat(t, n_samples))
        data["X1"] += list(X1)
        data["X2"] += list(X2)
        data["X3"] += list(X3)
        data["X4"] += list(X4)
        data["X5"] += list(X5)
        data["X6"] += list(X6)
        data["X7"] += list(X7)
    
    return data

def main():
    # Generate the data with 5000 samples per time step.
    # This provides a sizable dataset for each time period.
    data = generate_non_categorical_data(n_samples=5000)
    
    # Convert the dictionary to a pandas DataFrame, which makes further data manipulation easier.
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file.
    # This allows the data to be used later for analysis, visualization, or model testing.
    df.to_csv("non_categorical_gen_data.csv", index=False)
    print("Data saved to non_categorical_gen_data.csv")

# This ensures that main() runs when the script is executed directly.
if __name__ == "__main__":
    main()

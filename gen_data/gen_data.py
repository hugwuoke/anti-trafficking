import numpy as np

def generate_data(n_samples=5000):
    """
    Generate simulated data for time steps t=1,...,8.
    
    Returns:
        data: A dictionary where each key (t) maps to a NumPy array
              of shape (n_samples, 7) corresponding to the 7 attributes.
    """
    # Define the time steps (t = 1, 2, ..., 8)
    time_steps = np.arange(1, 9)
    data = {"t": sum([list(np.repeat(t, n_samples)) for t in time_steps], start=[]),
            "X1": [], "X2": [], "X3": [],
            "X4": [], "X5": [], "X6": [], "X7": []}

    for t in time_steps:
        # --- X1t: Baseline attribute (Bernoulli(0.2)) ---
        X1 = np.random.binomial(1, 0.2, n_samples)
        
        # --- X2t: Independent of X1 (Bernoulli(0.1)) ---
        X2 = np.random.binomial(1, 0.1, n_samples)
        
        # --- X3t: Highly dependent on X1 ---
        # For each sample, if X1 == 1 then sample Bernoulli(0.9); else 0.
        X3 = X1 * np.random.binomial(1, 0.9, n_samples)
        
        # --- X4t: Increasing dependence on X1 ---
        # Use probability 0.1 + 0.1*t (from 0.2 at t=1 to 0.9 at t=8).
        p4 = 0.1 + 0.1 * t
        X4 = X1 * np.random.binomial(1, p4, n_samples)
        
        # --- X5t: Decreasing dependence on X1 ---
        # Use probability 0.9 - 0.1*t (from 0.8 at t=1 to 0.1 at t=8).
        p5 = 0.9 - 0.1 * t
        X5 = X1 * np.random.binomial(1, p5, n_samples)
        
        # --- X6t: Shifting dependence on X1 ---
        # Use probability based on distance from t=4:
        # p6 = 0.1 + 0.2 * |t - 4|
        p6 = 0.1 + 0.2 * abs(t - 4)
        X6 = X1 * np.random.binomial(1, p6, n_samples)
        
        # --- X7t: Independent for t<8; dependent on X1 for t=8 ---
        if t < 8:
            X7 = np.random.binomial(1, 0.1, n_samples)
        else:  # t == 8
            X7 = X1 * np.random.binomial(1, 0.9, n_samples)
        
        # Append the generated data to the dictionary
        data["X1"] += list(X1)
        data["X2"] += list(X2)
        data["X3"] += list(X3)
        data["X4"] += list(X4)
        data["X5"] += list(X5)
        data["X6"] += list(X6)
        data["X7"] += list(X7)

    return data

def main():
    # Generate the data
    data = generate_data(n_samples=5000)
    # write the data as a csv
    import pandas as pd
    df = pd.DataFrame.from_dict(data)
    df.to_csv("simple_gen_data.csv", index=False)
    

if __name__ == '__main__':
    main()

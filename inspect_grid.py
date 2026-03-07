import pandapower.networks as nw
import pandas as pd

def inspect_grid(grid_case='case14'):
    # Load the requested IEEE bus system
    net = getattr(nw, grid_case)()
    
    print(f"--- PANDAPOWER {grid_case.upper()} TOPOLOGY ---")
    
    # Iterate through all attributes in the pandapower network
    for key, item in net.items():
        # Check if the item is a pandas DataFrame and is not empty
        if isinstance(item, pd.DataFrame) and not item.empty:
            print(f"\n=== Table: {key.upper()} ===")
            # Print the first 5 rows to inspect the features
            print(item.head())

if __name__ == "__main__":
    inspect_grid()
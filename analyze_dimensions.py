import json
import glob
import os
import pandas as pd

def analyze_results(directory, model_name):
    files = glob.glob(os.path.join(directory, "*.json"))
    records = []
    
    for f in files:
        with open(f, 'r', encoding='utf-8') as infile:
            try:
                data = json.load(infile)
                meta = data.get("meta", {})
                job = meta.get("job_title", "Unknown")
                soc = meta.get("onet_code", "Unknown")
                scores = meta.get("exposure_score", {})
                
                record = {
                    "Model": model_name,
                    "Job": job,
                    "SOC": soc,
                    "Overall": scores.get("overall", 0),
                }
                # Add individual dimension scores
                for dim, score in scores.get("dimensions", {}).items():
                    record[dim] = score
                    
                records.append(record)
            except Exception as e:
                print(f"Error parsing {f}: {e}")
                
    return pd.DataFrame(records)

def main():
    df_1_5b = pd.DataFrame()
    if os.path.exists("results"):
        df_1_5b = analyze_results("results", "qwen2.5:1.5b")
        
    df_7b = pd.DataFrame()
    if os.path.exists("results_7b"):
        df_7b = analyze_results("results_7b", "qwen2.5:7b")
        
    df_all = pd.concat([df_1_5b, df_7b], ignore_index=True)
    if df_all.empty:
        print("No results found yet.")
        return
        
    print("\n--- Dimension Decomposition Analysis ---")
    
    # Calculate the gap between Clarity and Domain Accuracy
    if "Clarity" in df_all.columns and "Domain Accuracy" in df_all.columns:
        df_all["Fluent_Flawed_Gap"] = df_all["Clarity"] - df_all["Domain Accuracy"]
        
        # Sort by gap to see which jobs have the biggest 'Fluent but Flawed' effect
        df_sorted = df_all.sort_values("Fluent_Flawed_Gap", ascending=False)
        print("\nTop 5 Jobs with highest 'Fluent but Flawed' Gap (Clarity > Domain Accuracy):")
        print(df_sorted[["Model", "Job", "Clarity", "Domain Accuracy", "Fluent_Flawed_Gap"]].head())
    
    print("\nAll Results summary:")
    print(df_all)
    df_all.to_csv("dimension_analysis_summary.csv", index=False)
    print("\nDetailed summary saved to dimension_analysis_summary.csv")

if __name__ == "__main__":
    main()

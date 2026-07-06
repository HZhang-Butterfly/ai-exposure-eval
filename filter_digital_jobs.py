#!/usr/bin/env python3
"""
filter_digital_jobs.py
----------------------
Reads Work Activities.xlsx and saves digital_jobs.csv.

A job is "digital" if it scores HIGH on computer work AND LOW on physical work.
All scores use Scale ID == 'IM' (Importance, 1-5 scale).
"""

# load data
import pandas as pd

df = pd.read_excel("Work Activities.xlsx")

df = df[df["Scale ID"] == "IM"]

print(f"  {len(df['O*NET-SOC Code'].unique())} unique occupations found")

# ── Step 2: Find jobs that are HIGH on computer/knowledge work

computers  = df[df["Element Name"] == "Working with Computers"]
processing = df[df["Element Name"] == "Processing Information"]
analyzing  = df[df["Element Name"] == "Analyzing Data or Information"]

high_computers  = set(computers [computers ["Data Value"] >= 4.0]["O*NET-SOC Code"])
high_processing = set(processing[processing["Data Value"] >= 3.0]["O*NET-SOC Code"])
high_analyzing  = set(analyzing [analyzing ["Data Value"] >= 3.0]["O*NET-SOC Code"])

print(f"  Working with Computers >= 4.0:        {len(high_computers)} jobs")
print(f"  Processing Information >= 3.0:        {len(high_processing)} jobs")
print(f"  Analyzing Data or Information >= 3.0: {len(high_analyzing)} jobs")

# ── Step 3: Find jobs that are LOW on physical work

physical   = df[df["Element Name"] == "Performing General Physical Activities"]
handling   = df[df["Element Name"] == "Handling and Moving Objects"]
vehicles   = df[df["Element Name"] == "Operating Vehicles, Mechanized Devices, or Equipment"]
machines   = df[df["Element Name"] == "Controlling Machines and Processes"]
repairing  = df[df["Element Name"] == "Repairing and Maintaining Mechanical Equipment"]

low_physical  = set(physical [physical ["Data Value"] <= 2.0]["O*NET-SOC Code"])
low_handling  = set(handling [handling ["Data Value"] <= 2.5]["O*NET-SOC Code"])
low_vehicles  = set(vehicles [vehicles ["Data Value"] <= 2.0]["O*NET-SOC Code"])
low_machines  = set(machines [machines ["Data Value"] <= 2.0]["O*NET-SOC Code"])
low_repairing = set(repairing[repairing["Data Value"] <= 2.5]["O*NET-SOC Code"])

print(f"  Physical Activities <= 2.0:           {len(low_physical)} jobs")
print(f"  Handling Objects <= 2.5:              {len(low_handling)} jobs")
print(f"  Operating Vehicles <= 2.0:            {len(low_vehicles)} jobs")
print(f"  Controlling Machines <= 2.0:          {len(low_machines)} jobs")
print(f"  Repairing Equipment <= 2.5:           {len(low_repairing)} jobs")

# ── Step 4: Keep only jobs that pass ALL conditions ───────────────────────────

digital_codes = (
    high_computers
    & high_processing
    & high_analyzing
    & low_physical
    & low_handling
    & low_vehicles
    & low_machines
    & low_repairing
)

print(f"\n {len(digital_codes)} digital jobs after all filters")

# ── Step 5: Build the output table ────────────────────────────────────────────

# One row per job (SOC code + title)
result = (
    df[df["O*NET-SOC Code"].isin(digital_codes)][["O*NET-SOC Code", "Title"]]
    .drop_duplicates(subset=["O*NET-SOC Code"])
    .sort_values("Title")
    .reset_index(drop=True)
)

# Add each activity score as an extra column (for transparency)
for element_df, col_name in [
    (computers,  "Working_with_Computers"),
    (processing, "Processing_Information"),
    (analyzing,  "Analyzing_Data"),
    (physical,   "Physical_Activities"),
    (handling,   "Handling_Objects"),
    (vehicles,   "Operating_Vehicles"),
    (machines,   "Controlling_Machines"),
    (repairing,  "Repairing_Equipment"),
]:
    scores = (
        element_df[["O*NET-SOC Code", "Data Value"]]
        .drop_duplicates(subset=["O*NET-SOC Code"])
        .rename(columns={"Data Value": col_name})
    )
    result = result.merge(scores, on="O*NET-SOC Code", how="left")


result.to_csv("digital_jobs.csv", index=False)
print(f"  Saved to digital_jobs.csv  ({len(result)} occupations)")
print()
print(result[["O*NET-SOC Code", "Title"]].head(15).to_string(index=False))

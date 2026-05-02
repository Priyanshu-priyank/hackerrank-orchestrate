import os
os.environ["CHROMA_TELEMETRY_IMPL"] = "None"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import argparse
import pandas as pd
from dotenv import load_dotenv

from retriever import Retriever
from classifier import classify
from agent import run_agent
from config import INPUT_CSV, OUTPUT_CSV, SAMPLE_CSV

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

def main():
    parser = argparse.ArgumentParser(description="Support Triage Agent")
    parser.add_argument("--sample", action="store_true", help="Run against sample data")
    args = parser.parse_args()

    csv_path = SAMPLE_CSV if args.sample else INPUT_CSV
    
    print(f"Reading input from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    # Normalize column names to lowercase to match code logic
    df.columns = [c.lower() for c in df.columns]
    # Replace NaN values with empty strings to avoid float comparison errors
    df = df.fillna("")
    
    # Initialize Retriever (loads model, connects to Chroma, indexes if needed)
    retriever = Retriever()
    
    results = []
    
    print(f"\nProcessing {len(df)} tickets...")
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        
        # 1. Classify (infer company, risk keywords)
        classification = classify(row_dict)
        
        # 2. Retrieve
        chunks, metas = retriever.query(
            text=row_dict.get('issue', ''),
            company=classification['company']
        )
        
        # 3. Agent (LLM)
        decision = run_agent(row_dict, chunks, metas, classification)
        
        # Format the output row matching the exact schema
        out_row = {
            "status": decision.status,
            "product_area": decision.product_area,
            "response": decision.response,
            "justification": decision.justification,
            "request_type": decision.request_type
        }
        results.append(out_row)
        
        print(f"[{idx}] {decision.status} | {decision.request_type} | {decision.product_area}")
        
    # Write to output.csv
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Results written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

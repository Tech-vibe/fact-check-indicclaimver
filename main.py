import sys
import os
import time

# Support both relative imports (package mode) and direct file execution
try:
    from .ranker import RankerPipeline
    from .utils import safe_read_json, atomic_write_json
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ranker import RankerPipeline
    from utils import safe_read_json, atomic_write_json


def main():
    # Ranker repository directory and parent project (Indic_Claim_Ver) directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Hardcoded default locations for Indic_Claim_Ver pipeline integration
    input_file = os.path.join(parent_dir, "Output", "Retrieved_Output")
    output_file = os.path.join(parent_dir, "Output", "Ranker_Output")

    # Optional total_count passed as command line argument: python Ranker/main.py [total_count]
    total_count = None
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        total_count = int(sys.argv[1])

    print("=" * 60)
    print(" IndicClaimVer 2026 - Subtask 2: Ranker Module")
    print(f" Input File  : {input_file}")
    print(f" Output File : {output_file}")
    if total_count:
        print(f" Total Claims: {total_count}")
    print("=" * 60 + "\n")

    # Load previously processed claims to allow resuming
    existing_output = safe_read_json(output_file)
    processed_ids = set()
    output_data = []

    for item in existing_output:
        item_id = item.get("ID") or item.get("id")
        if item_id:
            processed_ids.add(item_id)
            output_data.append(item)

    if output_data:
        print(f"[RANKER] Resuming from existing output -- {len(output_data)} claims already completed.")

    # Lazy load models
    pipeline = RankerPipeline()

    idle_ticks = 0
    max_idle_ticks = 5  # Stop after 5 consecutive idle ticks if stream is static

    print(f"[RANKER] Listening for claims in '{input_file}'...\n")

    while True:
        incoming_claims = safe_read_json(input_file)
        new_claims_found = False

        for claim_item in incoming_claims:
            claim_id = claim_item.get("ID") or claim_item.get("id")
            if not claim_id or claim_id in processed_ids:
                continue

            new_claims_found = True
            idle_ticks = 0

            # Process single claim
            result = pipeline.process_claim(claim_item)
            output_data.append(result)
            processed_ids.add(claim_id)

            # Atomic write to keep JSON valid for downstream LLM module
            atomic_write_json(output_file, output_data)

            curr_count = len(processed_ids)
            total_str = f"/{total_count}" if total_count else ""
            print(f"[RANKER] Processing claim {curr_count}{total_str} -- {claim_id}")

            # Check if total_count target reached
            if total_count and curr_count >= total_count:
                print(f"\n[RANKER] Reached target claim count ({total_count}). Processing complete!")
                return

        if not new_claims_found:
            # If total_count is known and reached, exit
            if total_count and len(processed_ids) >= total_count:
                break

            # If input file exists and has claims, but no new claims arrived after max_idle_ticks, exit
            if incoming_claims and len(processed_ids) == len(incoming_claims):
                idle_ticks += 1
                if idle_ticks >= max_idle_ticks:
                    print(f"\n[RANKER] All {len(processed_ids)} claims processed from static input file.")
                    break

            time.sleep(1.0)

    print("\n" + "=" * 60)
    print(" Ranker Reranking Completed Successfully!")
    print(f" Total Processed Claims : {len(output_data)}")
    print(f" Saved Output File      : {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

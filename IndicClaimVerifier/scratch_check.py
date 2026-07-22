import sys, json
sys.path.insert(0, r'D:\IndicClaimVerifier')

with open(r'D:\IndicClaimVerifier\topk_output.json', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    print(f"=== {item['ID']} ===")
    print(f"CLAIM: {item.get('Text','')}")
    print()

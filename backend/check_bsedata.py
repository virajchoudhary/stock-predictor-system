from bsedata.bse import BSE

try:
    b = BSE()
    print("BSE Connected.")
    
    # Check if we can get quotes
    q = b.getQuote("500325") # Reliance on BSE
    print(f"Quote Fetch: {q.get('currentValue')}")
    
    # Check for anything resembling Option Chain
    # bsedata documentation doesn't explicitly list options, inspecting attributes
    print(f"Available methods: {[m for m in dir(b) if not m.startswith('__')]}")

except Exception as e:
    print(f"BSE Data Error: {e}")

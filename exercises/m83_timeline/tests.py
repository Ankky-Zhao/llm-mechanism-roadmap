from datetime import date

assert applies("prohibited_practices", date(2025, 2, 2)) and not applies("prohibited_practices", date(2025, 2, 1))
assert applies("gpai_obligations", date(2026, 1, 1)) and not applies("high_risk_annex_iii", date(2026, 11, 25))
assert in_force(date(2024, 12, 31)) == []
assert in_force(date(2025, 3, 1)) == ["ai_literacy", "prohibited_practices"]
today = date(2026, 11, 25)
assert in_force(today) == ["ai_literacy", "gpai_obligations", "prohibited_practices", "transparency_art50"]
assert in_force(date(2028, 8, 2)) == sorted(APPLICATION)
assert days_until("high_risk_annex_iii", today) == (date(2027, 12, 2) - today).days == 372
assert days_until("gpai_obligations", today) == 0

assert max_fine("prohibited", 100_000_000) == 35_000_000                 # 7% would be 7 million; the fixed amount is higher
assert max_fine("prohibited", 2_000_000_000) == 140_000_000              # 7% of 2 billion
assert max_fine("other", 2_000_000_000) == 60_000_000 and max_fine("misleading", 2_000_000_000) == 20_000_000
assert max_fine("prohibited", 10_000_000, sme=True) == 700_000           # for an SME the lower amount
assert max_fine("other", 1_000_000_000, sme=True) == 15_000_000

print("m83_timeline: all tests passed")
print(f"  applying on {today}: {', '.join(in_force(today))}")
for o in ("high_risk_annex_iii", "high_risk_annex_i"):
    print(f"  {o}: from {APPLICATION[o]}, in {days_until(o, today)} days")

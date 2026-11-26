from datetime import date

# Dates of application as listed on the European Commission's AI Act page (last update 19 October 2026),
# which reflects the amendments of the Digital Omnibus on AI in force since 27 July 2026.
APPLICATION = {
    "prohibited_practices": date(2025, 2, 2),
    "ai_literacy": date(2025, 2, 2),
    "gpai_obligations": date(2025, 8, 2),
    "transparency_art50": date(2026, 8, 2),
    "high_risk_annex_iii": date(2027, 12, 2),
    "high_risk_annex_i": date(2028, 8, 2),
}


def applies(obligation, on):
    return on >= APPLICATION[obligation]


def in_force(on):
    return sorted(o for o in APPLICATION if applies(o, on))


def days_until(obligation, on):
    return max((APPLICATION[obligation] - on).days, 0)


def max_fine(kind, turnover_eur, sme=False):
    fixed, percent = {"prohibited": (35_000_000, 7), "other": (15_000_000, 3), "misleading": (7_500_000, 1)}[kind]
    share = turnover_eur * percent / 100                       # multiply first: 0.07 * x is not exact in floating point
    return min(fixed, share) if sme else max(fixed, share)

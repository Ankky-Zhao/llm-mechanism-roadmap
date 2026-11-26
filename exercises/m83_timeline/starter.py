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
    """True if the group of obligations applies on the date `on` (a datetime.date). The first day counts."""
    # TODO 1
    raise NotImplementedError


def in_force(on):
    """The names of all groups of obligations that apply on the date `on`, sorted alphabetically."""
    # TODO 2
    raise NotImplementedError


def days_until(obligation, on):
    """Days from `on` until the obligation starts to apply; 0 if it applies already."""
    # TODO 3
    raise NotImplementedError


def max_fine(kind, turnover_eur, sme=False):
    """The ceiling of an administrative fine under Article 99, in euros.

    kind "prohibited":  35 million or 7% of worldwide annual turnover     (prohibited practices)
    kind "other":       15 million or 3%                                  (most other obligations, including Article 50)
    kind "misleading":  7.5 million or 1%                                 (incorrect information to authorities)
    For an undertaking the HIGHER of the two amounts is the ceiling; for an SME the LOWER one.
    Compute the share as turnover_eur * percent / 100 so that round figures stay exact.
    """
    # TODO 4
    raise NotImplementedError

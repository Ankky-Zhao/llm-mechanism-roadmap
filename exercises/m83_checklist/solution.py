import re
from huggingface_hub import ModelCard

SECTIONS = {
    "intended_use": ["intended use", "uses", "use case"],
    "limitations": ["limitation", "bias", "risk"],
    "training_data": ["training data", "training", "dataset"],
    "evaluation": ["evaluation", "results", "performance"],
}


def parse_card(text):
    card = ModelCard(text)
    return card.data.to_dict(), card.text


def headings(body):
    return [h.strip().lower() for h in re.findall(r"^#+\s*(.+)$", body, flags=re.M)]


def check_card(text):
    meta, body = parse_card(text)
    heads = headings(body)
    report = {"license": bool(meta.get("license")), "language": bool(meta.get("language"))}
    for item, keywords in SECTIONS.items():
        report[item] = any(k in h for h in heads for k in keywords)
    return report


def completeness(report):
    return sum(report.values()) / len(report)

from huggingface_hub import HfApi

PERMISSIVE = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "cc-by-4.0", "isc", "unlicense"}
COPYLEFT = {"gpl-2.0", "gpl-3.0", "agpl-3.0", "lgpl-3.0", "mpl-2.0", "cc-by-sa-4.0"}


def classify_license(identifier, name=None):
    if not identifier or identifier == "unknown":
        return "unknown"
    ident = identifier.lower()
    label = (name or "").lower()
    if ident in PERMISSIVE:
        return "permissive"
    if ident in COPYLEFT:
        return "copyleft"
    if "-nc" in ident or "noncommercial" in ident or "research" in ident or "research" in label or label == "mrl":
        return "non-commercial"
    if "rail" in ident:
        return "use-restricted"
    return "custom"                                         # family licenses (llama*, gemma) and everything declared as "other"


def read_license(repo_id, api=None):
    info = (api or HfApi()).model_info(repo_id)
    card = info.card_data.to_dict() if info.card_data else {}
    identifier, name = card.get("license"), card.get("license_name")
    return {"repo": repo_id, "license": identifier, "name": name, "gated": bool(info.gated),
            "category": classify_license(identifier, name)}


def survey(repo_ids):
    api = HfApi()
    return [read_license(r, api) for r in repo_ids]

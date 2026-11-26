from huggingface_hub import HfApi

# The license of a model on the Hugging Face Hub is declared in the metadata of its model card:
#   license: apache-2.0                       an identifier from the Hub's list
#   license: other, license_name: mrl         a custom license, whose text is in the repository
# This exercise reads that metadata and sorts it into a few coarse categories. The category is a first
# orientation, not a legal judgement: the binding text is always the license itself.

PERMISSIVE = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "cc-by-4.0", "isc", "unlicense"}
COPYLEFT = {"gpl-2.0", "gpl-3.0", "agpl-3.0", "lgpl-3.0", "mpl-2.0", "cc-by-sa-4.0"}


def classify_license(identifier, name=None):
    """Sort a license into one of six categories. Check in this order:

      "unknown"         identifier is None, empty or "unknown"
      "permissive"      identifier in PERMISSIVE
      "copyleft"        identifier in COPYLEFT
      "non-commercial"  identifier contains "-nc", "noncommercial" or "research",
                        or the name contains "research", or the name is "mrl" (Mistral Research License)
      "use-restricted"  identifier contains "rail" (the OpenRAIL family)
      "custom"          everything else: family licenses such as llama3.3 or gemma, and "other"
    Compare in lower case. `name` may be None.
    """
    # TODO 1
    raise NotImplementedError


def read_license(repo_id, api=None):
    """Read the license metadata of one model repository.

    info = (api or HfApi()).model_info(repo_id)
    info.card_data.to_dict() holds the keys "license" and "license_name" (either may be missing,
    and card_data itself may be None). info.gated is False, "auto" or "manual".
    Returns {"repo", "license", "name", "gated" (a bool), "category"}.
    """
    # TODO 2
    raise NotImplementedError


def survey(repo_ids):
    """read_license for every repository, as a list of dicts."""
    # TODO 3
    raise NotImplementedError

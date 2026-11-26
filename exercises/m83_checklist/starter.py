import re
from huggingface_hub import ModelCard

# A checklist for the documentation of a model. It asks whether a model card states its license and
# language and has sections on intended use, limitations, training data and evaluation: the kinds of
# information that Mitchell et al. proposed for model cards and that downstream users need.
# It checks that a section EXISTS, not that its content is adequate, and it is not a test of legal compliance.

SECTIONS = {
    "intended_use": ["intended use", "uses", "use case"],
    "limitations": ["limitation", "bias", "risk"],
    "training_data": ["training data", "training", "dataset"],
    "evaluation": ["evaluation", "results", "performance"],
}


def parse_card(text):
    """Split a model card (the text of a README.md) into its metadata and its body.

    card = ModelCard(text);  card.data.to_dict() is the YAML metadata,  card.text the Markdown after it.
    Returns (metadata_dict, body).
    """
    # TODO 1
    raise NotImplementedError


def headings(body):
    """All Markdown headings of the body (lines starting with one or more #), lower-cased and stripped."""
    # TODO 2: re.findall(r"^#+\\s*(.+)$", body, flags=re.M)
    raise NotImplementedError


def check_card(text):
    """Returns a dict of six booleans:
      "license", "language":  the metadata field is present and not empty
      one entry per key of SECTIONS:  some heading contains one of the keywords of that section
    """
    # TODO 3
    raise NotImplementedError


def completeness(report):
    """The fraction of checklist items that are satisfied."""
    # TODO 4
    raise NotImplementedError

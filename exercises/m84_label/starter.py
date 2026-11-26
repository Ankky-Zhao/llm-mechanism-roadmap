# China's Measures for Labeling AI-Generated Synthetic Content (in force since 1 September 2025) distinguish
#   explicit labels: visible to the user, for text a notice at the start, the end or a suitable place in between;
#   implicit labels: written into the metadata of the file, containing the attribute of the content, the name
#                    or code of the service provider, and a content number.
# Platforms that distribute content have to check the metadata and tell their users what they found.
# This exercise builds a toy version of both sides. An item is a dict {"text": ..., "metadata": {...}}.
# The field names of the metadata are made up for this exercise; the real format is laid down in a national standard.

NOTICE = "（AI生成）"


def explicit_label(text, where="end", notice=NOTICE):
    """Add the visible notice at the start (where="start") or at the end (any other value) of the text."""
    # TODO 1
    raise NotImplementedError


def implicit_label(provider, content_id):
    """The metadata of a generated item: {"generated_by_ai": True, "provider": ..., "content_id": ...}"""
    # TODO 2
    raise NotImplementedError


def make_item(text, provider, content_id, where="end"):
    """What a generation service hands out: the text with its explicit label, plus the implicit label as metadata."""
    # TODO 3
    raise NotImplementedError


def platform_check(item, user_declared=False, notice=NOTICE):
    """The three cases a distribution platform has to tell apart, checked in this order:

      "generated"           the metadata carry generated_by_ai == True
      "possibly generated"  no such metadata, but the uploading user declares the content as generated
      "suspected"           neither of the above, but the visible notice occurs in the text
      "none"                otherwise
    The item may have no "metadata" key at all, or metadata that is None.
    """
    # TODO 4
    raise NotImplementedError


def display(item, user_declared=False):
    """The text as the platform shows it, with a prefix according to platform_check:
       "[AI-generated] ", "[possibly AI-generated] ", "[suspected AI-generated] " or no prefix."""
    # TODO 5
    raise NotImplementedError

NOTICE = "（AI生成）"


def explicit_label(text, where="end", notice=NOTICE):
    return notice + text if where == "start" else text + notice


def implicit_label(provider, content_id):
    return {"generated_by_ai": True, "provider": provider, "content_id": content_id}


def make_item(text, provider, content_id, where="end"):
    return {"text": explicit_label(text, where), "metadata": implicit_label(provider, content_id)}


def platform_check(item, user_declared=False, notice=NOTICE):
    meta = item.get("metadata") or {}
    if meta.get("generated_by_ai") is True:
        return "generated"                       # the implicit label is present
    if user_declared:
        return "possibly generated"              # no label, but the user says so
    if notice in item.get("text", ""):
        return "suspected"                       # no label, no declaration, but a visible trace
    return "none"


def display(item, user_declared=False):
    status = platform_check(item, user_declared)
    prefix = {"generated": "[AI-generated] ", "possibly generated": "[possibly AI-generated] ",
              "suspected": "[suspected AI-generated] ", "none": ""}[status]
    return prefix + item.get("text", "")

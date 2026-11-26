assert explicit_label("你好。") == "你好。（AI生成）" and explicit_label("你好。", "start") == "（AI生成）你好。"
assert explicit_label("Hi.", notice=" [AI]") == "Hi. [AI]"
assert implicit_label("example-llm", "c-0001") == {"generated_by_ai": True, "provider": "example-llm", "content_id": "c-0001"}

item = make_item("今天天气不错。", "example-llm", "c-0001")
assert item == {"text": "今天天气不错。（AI生成）", "metadata": {"generated_by_ai": True, "provider": "example-llm", "content_id": "c-0001"}}

# 1. The label is intact
assert platform_check(item) == "generated" and display(item).startswith("[AI-generated] ")

# 2. Someone copied the text into a new file: the metadata are gone, the visible notice is still there
copied = {"text": item["text"]}
assert platform_check(copied) == "suspected" and platform_check(copied, user_declared=True) == "possibly generated"
assert platform_check({"text": item["text"], "metadata": None}) == "suspected"

# 3. Someone removed the visible notice too. Nothing is left to detect, which is why removing labels is forbidden.
stripped = {"text": item["text"].replace(NOTICE, ""), "metadata": {}}
assert platform_check(stripped) == "none" and display(stripped) == "今天天气不错。"
assert platform_check(stripped, user_declared=True) == "possibly generated"

# 4. Only the visible notice was removed, the metadata survived
half = {"text": "今天天气不错。", "metadata": item["metadata"]}
assert platform_check(half) == "generated", "the metadata are checked first"

# 5. Ordinary content written by a person
human = {"text": "我自己写的一段话。", "metadata": {"author": "someone"}}
assert platform_check(human) == "none" and display(human) == "我自己写的一段话。"
assert display(copied) == "[suspected AI-generated] 今天天气不错。（AI生成）"
assert display(stripped, user_declared=True) == "[possibly AI-generated] 今天天气不错。"

print("m84_label: all tests passed")
for name, it, declared in (("intact", item, False), ("metadata lost", copied, False), ("all labels removed", stripped, False), ("removed, user declares", stripped, True)):
    print(f"  {name:24s} -> {platform_check(it, declared)}")

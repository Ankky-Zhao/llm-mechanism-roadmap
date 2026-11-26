import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Luhn
assert luhn_valid("4539 1488 0343 6467") and luhn_valid("4111111111111111")       # standard test numbers, not real cards
assert not luhn_valid("4539 1488 0343 6468") and not luhn_valid("1234 5678 9012 3456") and not luhn_valid("79927398713")

# 2. Redaction. Every name, address and number below is invented.
text = ("Please contact Mara Lindqvist at mara.lindqvist@example.org or +49 89 1234 5678. "
        "Her IBAN is DE44 5001 0517 5407 3249 31 and her card number is 4539 1488 0343 6467. "
        "Order 1234 5678 9012 3456 was shipped to Quay Lane 7 on 12 March.")
out = redact(text)
assert "mara.lindqvist@example.org" not in out and "[EMAIL]" in out
assert "+49 89 1234 5678" not in out and "[PHONE]" in out
assert "DE44" not in out and "[IBAN]" in out
assert "4539 1488 0343 6467" not in out and "[CARD]" in out
assert "1234 5678 9012 3456" in out, "an order number that fails the checksum is not a card number"
assert "Mara Lindqvist" in out and "Quay Lane 7" in out, "patterns do not find names or addresses"
assert redact("Nothing personal here, 3 apples and 12 pears.") == "Nothing personal here, 3 apples and 12 pears."

# 3. The probe on GPT-2
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
OCCUPATIONS = ["nurse", "librarian", "teacher", "receptionist", "secretary", "doctor", "carpenter", "mechanic", "pilot", "engineer"]
gaps = {o: pronoun_gap(model, tok, o) for o in OCCUPATIONS}
neutral = pronoun_gap(model, tok, "person")
with torch.no_grad():
    ref = torch.log_softmax(model(tok("The nurse said that", return_tensors="pt").input_ids).logits[0, -1], -1)
assert abs(gaps["nurse"] - (ref[tok(" he").input_ids[0]] - ref[tok(" she").input_ids[0]]).item()) < 1e-4
assert gaps["engineer"] > gaps["nurse"] + 2, "GPT-2 associates these two occupations with different pronouns"
assert gaps["nurse"] < 0 < gaps["mechanic"]
assert max(gaps.values()) - min(gaps.values()) > 3

print("m82_pii: all tests passed")
print("  redacted:", out)
print("  log p(he) - log p(she) after 'The <occupation> said that'   (positive: 'he' more likely)")
for o, g in sorted(gaps.items(), key=lambda kv: kv[1]):
    print(f"    {o:13s} {g:+.2f}  {'#' * int(abs(g) * 8)}")
print(f"    {'person':13s} {neutral:+.2f}")

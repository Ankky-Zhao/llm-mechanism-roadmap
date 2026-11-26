import tiktoken

enc = tiktoken.get_encoding("o200k_base")
assert count_tokens(enc, "Hello world") == 2
assert count_tokens(enc, "北京是中国的首都") == 6

# three turns: user 10, 20, 30 tokens; assistant 100 each; system prompt 50
i, o = chat_billed_tokens([10, 20, 30], [100, 100, 100], system_len=50)
assert o == 300
assert i == (50 + 10) + (50 + 10 + 100 + 20) + (50 + 10 + 100 + 20 + 100 + 30), f"got {i}"

# the history is sent again every turn, so the input grows quadratically with the number of turns
i10, _ = chat_billed_tokens([50] * 10, [200] * 10)
i20, _ = chat_billed_tokens([50] * 20, [200] * 20)
assert 3.5 < i20 / i10 < 4.2, f"twice as many turns should cost about four times the input tokens, got {i20 / i10:.2f}"
assert i20 > 10 * sum([200] * 20), "after 20 turns the re-sent history far outweighs the generated text"

assert abs(cost(1_000_000, 0, 3.0, 15.0) - 3.0) < 1e-12
assert abs(cost(250_000, 20_000, 3.0, 15.0) - (0.75 + 0.30)) < 1e-12

assert fit_to_window([50, 100, 100, 100], 400) == [0, 1, 2, 3], "everything fits"
assert fit_to_window([50, 100, 100, 100], 300) == [0, 2, 3], "drop the oldest message after the system prompt"
assert fit_to_window([50, 100, 100, 100], 160) == [0, 3]
assert fit_to_window([50, 100, 100, 100], 200, keep_first=False) == [2, 3]
assert fit_to_window([50, 100, 100, 100], 10) == [0], "the system prompt stays even if nothing else fits"
print("m32_budget: all tests passed")
print(f"  10 turns: {i10:,} input tokens, 20 turns: {i20:,} input tokens")

Real Academic Content Corpus (v2 - text only)
Source: Shayla's own course material (FDS, Information Security, Spanish), extracted from her PPTX/PDF slide decks.

30 documents, ~37,600 words total, plain text only: no SQL queries, no router/CLI config commands, no code, no formulas, no arrow/bracket diagram notation, no URLs.
corpus/ : the 30 cleaned documents, named by topic.
gold_qa.json : 47 single-hop fact questions with verified answers.

Note: a few heavily code-based slides (SQL window functions, ABAC/ABE examples) needed real rewriting, not just stripping, to stay coherent once all code and worked numeric examples were removed. Everything else is lightly cleaned real extracted text.

No multi-hop entity graph here since the content spans three unrelated subjects. Use for factual accuracy and hallucination-resistance testing, not hop-depth testing.

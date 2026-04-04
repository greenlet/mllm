"""SQuAD v1.1 dataset loader for QnA MixedDecoder training.

HuggingFace: rajpurkar/squad
Splits: train (~87K), validation (~10K)
Columns: id, title, context, question, answers (text: list[str], answer_start: list[int])
Same format as SQuAD v2 but answerable-only.
Needs deduplication against SQuAD v2 by id (v2 contains all v1 answerable questions).
Unanswerable: no
Multi-turn: no
"""

from mllm.data.qna.dataset import QnaBaseDataset

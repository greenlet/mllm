"""SQuAD v2 dataset loader for QnA MixedDecoder training.

HuggingFace: rajpurkar/squad_v2
Splits: train, validation
Columns: id, title, context, question, answers (text: list[str], answer_start: list[int])
Unanswerable: yes — answers['text'] is empty (~33% of data)
Multi-turn: no
"""

from mllm.data.qna.dataset import QnaBaseDataset

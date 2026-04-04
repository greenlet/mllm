"""NewsQA dataset loader for QnA MixedDecoder training.

HuggingFace: lucadiliello/newsqa
Splits: train (~74K), validation (~4K)
Columns: context, question, answers (list[str]), key, labels
Context: full CNN news article
Answer: answers[0]
Unanswerable: no
Multi-turn: no — multiple questions per article
"""

from mllm.data.qna.dataset import QnaBaseDataset

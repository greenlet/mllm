"""AdversarialQA dataset loader for QnA MixedDecoder training.

HuggingFace: adversarial_qa (config: adversarialQA)
Splits: train (~30K), validation (~3K), test (~3K)
Columns: id, title, context, question, answers (text: list[str], answer_start: list[int])
SQuAD-identical format. Adversarially collected to be hard but answerable.
Unanswerable: no
Multi-turn: no
"""

from mllm.data.qna.dataset import QnaBaseDataset

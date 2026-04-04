"""MRQA dataset loader for QnA MixedDecoder training.

HuggingFace: mrqa
Splits: train (~517K), validation (~58K), test (~10K)
Columns: subset, context, question, answers (list[str]), detected_answers
Train subsets: SQuAD, NewsQA, TriviaQA-web, SearchQA, HotpotQA, NaturalQuestionsShort
Only SearchQA and HotpotQA subsets are kept to avoid overlap with standalone datasets.
Unanswerable: no
Multi-turn: no
"""

from mllm.data.qna.dataset import QnaBaseDataset

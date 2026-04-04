"""Natural Questions dataset loader for QnA MixedDecoder training.

HuggingFace: google-research-datasets/natural_questions
Splits: train (~307K), validation (~7.8K)
Columns: id, question (dict with 'text'), document (tokenized HTML), annotations (short_answers, long_answer, yes_no_answer)
Context: extracted from document HTML tokens; long_answer span used as focused context
Answer: short_answer text extracted from token spans
Unanswerable: yes — no short answer exists (~36% of val)
Multi-turn: no
"""

from mllm.data.qna.dataset import QnaBaseDataset

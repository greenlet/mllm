"""CoQA dataset loader for QnA MixedDecoder training.

HuggingFace: stanfordnlp/coqa
Splits: train, validation
Columns: source (domain), story, questions (list[str]),
         answers (input_text, answer_start, answer_end per turn)
Context: story (passage from 5 domains: race, cnn, wikipedia, gutenberg, mctest)
Answer: answers['input_text'] — abstractive human-written answer (preferred over extractive span)
Unanswerable: yes — input_text == 'unknown' (~1.3% of turns)
Multi-turn: yes — each row is a multi-turn dialogue
"""

from mllm.data.qna.dataset import QnaBaseDataset

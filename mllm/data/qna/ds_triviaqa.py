"""TriviaQA dataset loader for QnA MixedDecoder training.

HuggingFace: trivia_qa (config: rc)
Splits: train, validation, test
Columns: question, question_id, answer (dict: value, aliases), entity_pages, search_results
Context: entity_pages['wiki_context'][0] or search_results['search_context'][0] — very long passages
Answer: answer['value']
Unanswerable: no
Multi-turn: no
"""

from mllm.data.qna.dataset import QnaBaseDataset

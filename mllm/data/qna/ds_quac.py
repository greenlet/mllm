"""QuAC dataset loader for QnA MixedDecoder training.

HuggingFace: quac (revision: refs/convert/parquet)
Splits: train, validation
Columns: dialogue_id, wikipedia_page_title, background, section_title, context,
         questions (list[str]), orig_answers (texts, answer_starts), followups, yesnos, turn_ids
Context: Wikipedia section passage shared across all turns
Answer: orig_answers['texts'] per turn; 'CANNOTANSWER' for unanswerable turns (~17%)
Unanswerable: yes
Multi-turn: yes — each row is a full dialogue with multiple Q/A turns
"""

from mllm.data.qna.dataset import QnaBaseDataset

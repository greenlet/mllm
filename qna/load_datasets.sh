#!/bin/bash
set -euo pipefail

data_path=./data

python -m qna.load_datasets \
    --data-path "$data_path" \
    --qna-datasets \
        squad_v2 \
        natural_questions \
        triviaqa \
        newsqa \
        mrqa \
        adversarialqa \
        quac \
        coqa

# python -m qna.load_datasets --data-path Q:/data --qna-datasets  squad_v2 natural_questions triviaqa newsqa mrqa adversarialqa quac coqa


import re

from app.core_paths import ML_RESULTS_DIR

EXAMPLES_PATH = ML_RESULTS_DIR / 'qualitative_examples.txt'


def parse_qualitative_examples():
    text = EXAMPLES_PATH.read_text(encoding='utf-8')
    pattern = re.compile(
        r'QUERY\s+(?P<index>\d+):\s+"(?P<query>.*?)".*?GROUND TRUTH DUPLICATES \(\d+\):\s+\[RELEVANT\]\s+"(?P<ground_truth>.*?)".*?--- STAGE 1:.*?---\sSTAGE 2:.*?Duplicate was at rank\s(?P<before>\d+)\safter retrieval, moved to rank\s(?P<after>\d+)\safter reranking\s+>>\s(?P<summary>.*?)(?=\n={10,}|\Z)',
        re.DOTALL,
    )
    stage_rank_pattern = re.compile(r'Rank\s+(\d+):\s+\[(.*?)\]\s+(\[(?:DUPLICATE|x)\])\s+"(.*?)"')
    stage2_section_pattern = re.compile(r'--- STAGE 2: Cross-Encoder Reranking ---\n(?P<body>.*?)(?:\n\s*>> Duplicate was at rank)', re.DOTALL)
    stage1_section_pattern = re.compile(r'--- STAGE 1:.*?---\n(?P<body>.*?)(?:\n--- STAGE 2:)', re.DOTALL)

    examples = []
    for match in pattern.finditer(text):
        chunk = match.group(0)
        stage1_body_match = stage1_section_pattern.search(chunk)
        stage2_body_match = stage2_section_pattern.search(chunk)
        stage1_body = stage1_body_match.group('body') if stage1_body_match else ''
        stage2_body = stage2_body_match.group('body') if stage2_body_match else ''
        stage1 = []
        stage2 = []
        for rank, score, dup_flag, item_text in stage_rank_pattern.findall(stage1_body):
            stage1.append({
                'rank': int(rank),
                'score': float(score),
                'text': item_text,
                'is_duplicate': dup_flag == '[DUPLICATE]',
            })
        for rank, score, dup_flag, item_text in stage_rank_pattern.findall(stage2_body):
            stage2.append({
                'rank': int(rank),
                'score': float(score),
                'text': item_text,
                'is_duplicate': dup_flag == '[DUPLICATE]',
            })
        examples.append({
            'index': int(match.group('index')),
            'query': match.group('query'),
            'ground_truth': match.group('ground_truth'),
            'before_rank': int(match.group('before')),
            'after_rank': int(match.group('after')),
            'improvement': int(match.group('before')) - int(match.group('after')),
            'summary': match.group('summary').strip(),
            'stage1': stage1,
            'stage2': stage2,
        })
    return examples

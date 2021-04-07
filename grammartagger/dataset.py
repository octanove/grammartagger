import json
from typing import Any, Dict, List, Tuple, Optional

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, ListField, SpanField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
from allennlp_models.rc.dataset_readers.utils import char_span_to_token_span


@DatasetReader.register("grammartagger")
class GrammarTaggerDatasetReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = 128
    ) -> None:
        super().__init__()

        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(
        self,
        sentence: str,
        char_spans: List[Tuple[int, int]] = None,
        labels: List[str] = None,
        level: str = None
    ) -> Instance:
        tokens = self._tokenizer.tokenize(sentence)
        token_strs = [str(t) for t in tokens]
        token_offsets = [(token.idx or 0, (token.idx or 0) + len(token.text)) for token in tokens]
        if char_spans is None:
            char_spans = []
        if labels is None:
            labels = []

        text_field = TextField(tokens, self._token_indexers)
        span_fields = []
        span_labels = []
        span_indices = set()
        for (char_span_start, char_span_end), label in zip(char_spans, labels):
            (span_start, span_end), error = char_span_to_token_span(token_offsets, (char_span_start, char_span_end))
            # print('\t',  span_end - span_start, sentence[char_span_start:char_span_end], tokens[span_start:span_end+1], error, label)
            if span_start > self.max_tokens or span_end > self.max_tokens:
                continue
            span_indices.add((span_start, span_end))
            span_fields.append(SpanField(span_start, span_end, text_field))
            span_labels.append(label)

        tokens = tokens[:self.max_tokens]
        # add negative spans
        max_span_len = 30
        for i in range(len(tokens)):
            for j in range(i, len(tokens)):
                if (i, j) not in span_indices and j - i + 1 <= max_span_len:
                    if '；' in token_strs[i:j+1]:
                        continue
                    span_fields.append(SpanField(i, j, text_field))
                    span_labels.append('NONE')

        if len(span_labels) > 1000:
            print(sentence)

        span_fields = ListField(span_fields)
        span_labels = SequenceLabelField(span_labels, span_fields)
        fields = {
            'tokens': text_field,
            'spans': span_fields,
            'span_labels': span_labels,
        }

        if level is not None:
            fields['level'] = LabelField(level, label_namespace='levels')

        return Instance(fields)


    def read(self, file_path):
        with open(file_path) as f:
            for line in f:
                data = json.loads(line)
                sentence = data['sentence']
                gis = [(start_index, end_index) for start_index, end_index, _ in data['gis']]
                labels = [label for _, _, label in data['gis']]
                level = data['level']

                yield self.text_to_instance(sentence, gis, labels, level)


if __name__ == '__main__':
    import logging
    logger = logging.getLogger("spacy")
    logger.setLevel(logging.ERROR)

    reader = GrammarTaggerDatasetReader()
    for instance in reader.read('data/en/train.jsonl'):
        print(instance)

import sys
import json

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
import grammartagger


def main():
    cuda_device = 0
    archive_file = sys.argv[1]
    predictor_name = 'grammartagger'

    archive = load_archive(
        archive_file=archive_file,
        cuda_device=cuda_device
    )
    level_id_to_label = archive.model.vocab.get_index_to_token_vocabulary('levels')

    predictor = Predictor.from_archive(archive, predictor_name=predictor_name)

    for line in sys.stdin:
        pred = predictor.predict_json({'sentence': line})
        level_probs = {}
        for i, prob in enumerate(pred['level_probs']):
            level_label = level_id_to_label[i]
            if level_label in {'@@PADDING@@', '@@UNKNOWN@@'}:
                continue
            level_probs[level_label] = prob
        pred['level_probs'] = level_probs
        print(json.dumps(pred, ensure_ascii=False))

if __name__ == '__main__':
    main()

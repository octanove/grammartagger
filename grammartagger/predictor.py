import torch

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register("grammartagger")
class GrammarTaggerPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self.vocab = model.vocab

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        pred = torch.max(torch.tensor(output_dict['class_probs']), dim=-1)[1]
        pred = pred.tolist()

        result = {
            'spans': []
        }
        tokens = instance.fields['tokens'].tokens
        for span, label_id in zip(instance.fields['spans'], pred):
            label = self.vocab.get_token_from_index(label_id, namespace='labels')
            if label == 'NONE':
                continue
            sub_tokens = [str(t) for t in tokens[span.span_start:span.span_end+1]]
            result['spans'].append({'span': [span.span_start, span.span_end], 'tokens': sub_tokens, 'label': label})
        
        result['tokens'] = [str(t) for t in tokens]
        result['level_probs'] = output_dict['level_probs']

        return result

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)

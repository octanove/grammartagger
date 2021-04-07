from overrides import overrides
from typing import Dict

import torch
from torch.nn.modules.linear import Linear

from allennlp.common.util import nan_safe_tensor_divide
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, FeedForward
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, F1Measure


@Model.register("grammartagger")
class GrammarTaggerModel(Model):
    def __init__(self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        span_extractor: SpanExtractor,
        encoder: Seq2SeqEncoder,
        feedforward: FeedForward = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        alpha: float = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.text_field_embedder = text_field_embedder
        self.span_extractor = span_extractor
        num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = encoder
        self.feedforward_layer = TimeDistributed(feedforward) if feedforward else None
        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = span_extractor.get_output_dim()
        self.tag_projection_layer = TimeDistributed(Linear(output_dim, num_classes))

        num_levels = self.vocab.get_vocab_size("levels")
        self.level_projection_layer = Linear(encoder.get_output_dim(), num_levels)
        self.level_loss = torch.nn.CrossEntropyLoss()
        self.alpha = alpha

        self.none_label_id = vocab.get_token_index('NONE', namespace='labels')
        self.tag_accuracy = CategoricalAccuracy()
        self.level_accuracy = CategoricalAccuracy()
        self.labeled_f1 = FBetaMeasure(beta=1.0, average='macro')
        self.unlabeled_f1 = F1Measure(positive_label=1)
        initializer(self)

    @overrides
    def forward(
        self,
        tokens: TextFieldTensors,
        spans: torch.LongTensor,
        span_labels: torch.LongTensor = None,
        level: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        embedded_text_input = self.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1)
        encoded_text = self.encoder(embedded_text_input, mask)
        span_representations = self.span_extractor(encoded_text, spans, mask, span_mask)
        if self.feedforward_layer is not None:
            span_representations = self.feedforward_layer(span_representations)
        logits = self.tag_projection_layer(span_representations)
        class_probs = masked_softmax(logits, span_mask.unsqueeze(-1))

        level_logits = self.level_projection_layer(encoded_text[:, 0, :])
        level_probs = torch.nn.functional.softmax(level_logits, dim=-1)

        output_dict = {
            "class_probs": class_probs,
            "level_probs": level_probs,
            "spans": spans
        }
        if span_labels is not None:
            loss = sequence_cross_entropy_with_logits(logits, span_labels, span_mask)
            self.tag_accuracy(class_probs, span_labels, span_mask)
            self.labeled_f1(class_probs, span_labels, span_mask)
            none_probs = (class_probs.argmax(dim=-1) == self.none_label_id).float().unsqueeze(-1)
            binary_probs = torch.cat([none_probs, 1. - none_probs], dim=-1)
            binary_labels = (span_labels != self.none_label_id).int()

            self.unlabeled_f1(binary_probs, binary_labels, span_mask)

            if level is not None and self.alpha is not None:
                level_loss = self.level_loss(level_logits, level)
                loss += self.alpha * level_loss
                self.level_accuracy(level_probs, level)

            output_dict["loss"] = loss

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "tag_accuracy": self.tag_accuracy.get_metric(reset=reset),
            "level_accuracy": self.level_accuracy.get_metric(reset=reset)
        }

        # labeled prec, rec, F1
        tp_neg = self.labeled_f1._true_positive_sum[self.none_label_id]
        tp_pos = self.labeled_f1._true_positive_sum.sum() - tp_neg
        pred_neg = self.labeled_f1._pred_sum[self.none_label_id]
        pred_pos = self.labeled_f1._pred_sum.sum() - pred_neg
        true_neg = self.labeled_f1._true_sum[self.none_label_id]
        true_pos = self.labeled_f1._true_sum.sum() - true_neg

        labeled_prec = nan_safe_tensor_divide(tp_pos, pred_pos)
        labeled_rec = nan_safe_tensor_divide(tp_pos, true_pos)
        labeled_f1 = nan_safe_tensor_divide(2. * labeled_prec * labeled_rec, labeled_prec + labeled_rec)

        labeled_macro_f1 = self.labeled_f1.get_metric(reset=reset)

        unlabeled_metrics = self.unlabeled_f1.get_metric(reset=reset)

        metrics.update({
            'labeled_prec': labeled_prec.item(),
            'labeled_rec': labeled_rec.item(),
            'labeled_f1': labeled_f1.item(),
            'labeled_macro_prec': labeled_macro_f1['precision'],
            'labeled_macro_rec': labeled_macro_f1['recall'],
            'labeled_macro_f1': labeled_macro_f1['fscore'],
            'unlabeled_prec': unlabeled_metrics['precision'],
            'unlabeled_rec': unlabeled_metrics['recall'],
            'unlabeled_f1': unlabeled_metrics['f1']
        })

        return metrics

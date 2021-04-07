local bert_model = "bert-base-uncased";

{
    "dataset_reader": {
        "type": "grammartagger",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
            }
        }
    },

    "train_data_path": "data/en/train.jsonl",
    "validation_data_path": "data/en/valid.jsonl",
    "model": {
        "type": "grammartagger",
        "text_field_embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model
                }
            }
        },
        "encoder": {
            "type": "pass_through",
            "input_dim": 768
        },
        "span_extractor": {
            "type": "endpoint",
            "combination": "x,y,x-y",
            "input_dim": 768
        },
        "initializer": {
            "regexes": [
                ["tag_projection_layer.*weight", {"type": "xavier_normal"}],
                ["level_projection_layer.*weight", {"type": "xavier_normal"}],
            ],
        },
        "alpha": null
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size" : 50
        }
    },
    "trainer": {
        "num_epochs": 100,
        "validation_metric": "+labeled_f1",
        "optimizer": {
            "type": "adam",
            "lr": 5.0e-5,
        },
        "cuda_device": 0
    }
}

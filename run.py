import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
import evaluate
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json

NUM_PREPROCESSING_WORKERS = 2


def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    # --do_eval
    # --per_device_train_batch_size <int, default=8>
    # --num_train_epochs <float, default=3.0>
    # --output_dir <path>

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection and preprocessing
    if args.dataset == "snli:anli":
        print("Loading SNLI and ANLI datasets...")
        snli = datasets.load_dataset("snli")
        anli = datasets.load_dataset("anli")

        # Remove invalid labels (-1) from SNLI
        snli = snli.filter(lambda ex: ex['label'] != -1)

        # Concatenate ANLI rounds
        anli_train = datasets.concatenate_datasets(
            [anli["train_r1"], anli["train_r2"], anli["train_r3"]]
        )
        anli_validation = datasets.concatenate_datasets(
            [anli["dev_r1"], anli["dev_r2"], anli["dev_r3"]]
        )

        # Merge SNLI and ANLI datasets
        train_dataset = datasets.concatenate_datasets([snli["train"], anli_train])
        eval_dataset = datasets.concatenate_datasets([snli["validation"], anli_validation])
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else default_datasets[args.task]
        dataset = datasets.load_dataset(*dataset_id)
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']

    if args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
    else:
        raise ValueError(f"Unrecognized task name: {args.task}")

    print("Preprocessing data... (this might take some time)")
    train_dataset = train_dataset.map(
        prepare_train_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        prepare_eval_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=eval_dataset.column_names
    )

    # Model and training setup
    model_classes = {'qa': AutoModelForQuestionAnswering, 'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    model = model_class.from_pretrained(args.model, num_labels=3 if args.task == 'nli' else None)

    trainer_class = Trainer if args.task == 'nli' else QuestionAnsweringTrainer

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy if args.task == 'nli' else None
    )

    # Train and evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # Save evaluation results
        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')


if __name__ == "__main__":
    main()

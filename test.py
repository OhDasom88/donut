"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from donut import DonutModel, JSONParseEvaluator, load_json, save_json


def test(args):
    pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")

    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    predictions = []
    ground_truths = []
    accs = []

    evaluator = JSONParseEvaluator()
    # dataset = load_dataset(args.dataset_name_or_path, split=args.split)
    from datasets import Dataset, DatasetDict, Image, Features, Value
    import pyarrow as pa
    dataset_dir = "/data/datasets/naver-clova-ix/cord-v2/naver-clova-ix___cord-v2/naver-clova-ix--cord-v2-1b6a08e905758c38/0.0.0/e58c486e4bad3c9cf8d969f920449d1103bbdf069a7150db2cf96c695aeca990"
    partial_features = Features({
        "image": Image(),  # 이미지 경로 복원
        "ground_truth": Value("string"),
        # 기타 필드...
    })

    # ↓ Dataset(pa_table=...) 대신 직접 생성
    def load_arrow_table(path):
        with pa.memory_map(path, "r") as source:
            reader = pa.ipc.RecordBatchStreamReader(source)
            return reader.read_all()
    if args.split == "train":
        train_table = pa.concat_tables([
            load_arrow_table(f"{dataset_dir}/cord-v2-train-00000-of-00002.arrow"),
            load_arrow_table(f"{dataset_dir}/cord-v2-train-00001-of-00002.arrow"),
        ])
        # 처음 30개만 잘라내서 테스트
        train_table = train_table.slice(0, 30)
        # train_table = train_table.slice(0, 10)
        
        # self.dataset = Dataset.from_dict(train_table.to_pydict())
        dataset = Dataset.from_dict(train_table.to_pydict(), features=partial_features)
        # self.dataset = Dataset(pa_table=train_table)
    elif args.split == "validation":
        validation_table = load_arrow_table(f"{dataset_dir}/cord-v2-validation.arrow")
        # 처음 10개만 잘라내서 테스트
        validation_table = validation_table.slice(0, 10)
        dataset = Dataset.from_dict(validation_table.to_pydict(), features=partial_features)
    elif args.split == "test":
        dataset = Dataset.from_dict(load_arrow_table(f"{dataset_dir}/cord-v2-test.arrow").to_pydict(), features=partial_features)
    else:
        raise ValueError(f"Invalid split: {args.split}")

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        ground_truth = json.loads(sample["ground_truth"])

        if args.task_name == "docvqa":
            output = pretrained_model.inference(
                image=sample["image"],
                prompt=f"<s_{args.task_name}><s_question>{ground_truth['gt_parses'][0]['question'].lower()}</s_question><s_answer>",
            )["predictions"][0]
        else:
            output = pretrained_model.inference(image=sample["image"], prompt=f"<s_{args.task_name}>")["predictions"][0]

        if args.task_name == "rvlcdip":
            gt = ground_truth["gt_parse"]
            score = float(output["class"] == gt["class"])
        elif args.task_name == "docvqa":
            # Note: we evaluated the model on the official website.
            # In this script, an exact-match based score will be returned instead
            gt = ground_truth["gt_parses"]
            answers = set([qa_parse["answer"] for qa_parse in gt])
            score = float(output["answer"] in answers)
        else:
            gt = ground_truth["gt_parse"]
            score = evaluator.cal_acc(output, gt)

        accs.append(score)

        predictions.append(output)
        ground_truths.append(gt)

    scores = {
        "ted_accuracies": accs,
        "ted_accuracy": np.mean(accs),
        "f1_accuracy": evaluator.cal_f1(predictions, ground_truths),
    }
    print(
        f"Total number of samples: {len(accs)}, Tree Edit Distance (TED) based accuracy score: {scores['ted_accuracy']}, F1 accuracy score: {scores['f1_accuracy']}"
    )

    if args.save_path:
        scores["predictions"] = predictions
        scores["ground_truths"] = ground_truths
        save_json(args.save_path, scores)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path", 
        default="/home/dasom/donut/result/train_cord/test_experiment",
        type=str
    )
    parser.add_argument(
        "--dataset_name_or_path", 
        default="/data/datasets/naver-clova-ix/cord-v2/naver-clova-ix___cord-v2/naver-clova-ix--cord-v2-1b6a08e905758c38/0.0.0/e58c486e4bad3c9cf8d969f920449d1103bbdf069a7150db2cf96c695aeca990",
        type=str
    )
    
    parser.add_argument(
        "--split", 
        type=str, 
        # default="test"
        default="train"
    )
    parser.add_argument(
        "--task_name", 
        type=str, 
        # default=None
        default="cord"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default=f'result/train_cord_train_cord-v2.json'
    )
    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)

    predictions = test(args)

    predictions

"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import torch
import zss
from datasets import load_dataset
from nltk import edit_distance
from torch.utils.data import Dataset
from transformers.modeling_utils import PreTrainedModel
from zss import Node





def save_json(write_path: Union[str, bytes, os.PathLike], save_obj: Any):
    with open(write_path, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, indent=4, ensure_ascii=False)


def load_json(json_path: Union[str, bytes, os.PathLike]):
    with open(json_path, "r") as f:
        return json.load(f)


class DonutDataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string)

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        donut_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key

 
        # self.dataset = load_dataset(dataset_name_or_path, split=self.split)
 
        from datasets import Dataset, DatasetDict, Image, Features, Value
        import pyarrow as pa


        # dataset_dir = "/data/datasets/naver-clova-ix/cord-v2/naver-clova-ix___cord-v2/naver-clova-ix--cord-v2-1b6a08e905758c38/0.0.0/e58c486e4bad3c9cf8d969f920449d1103bbdf069a7150db2cf96c695aeca990"
        # dataset_dir = "/data/datasets/naver-clova-ix/cord-v2/naver-clova-ix___cord-v2/naver-clova-ix--cord-v2-1b6a08e905758c38/0.0.0/e58c486e4bad3c9cf8d969f920449d1103bbdf069a7150db2cf96c695aeca990"
        # partial_features = Features({
        #     "image": Image(),  # 이미지 경로 복원
        #     "ground_truth": Value("string"),
        #     # 기타 필드...
        # })

        # # ↓ Dataset(pa_table=...) 대신 직접 생성
        # def load_arrow_table(path):
        #     with pa.memory_map(path, "r") as source:
        #         reader = pa.ipc.RecordBatchStreamReader(source)
        #         return reader.read_all()

        # if self.split == "train":
        #     # train_table = pa.concat_tables([
        #     #     load_arrow_table(f"{dataset_dir}/cord-v2-train-00000-of-00002.arrow"),
        #     #     load_arrow_table(f"{dataset_dir}/cord-v2-train-00001-of-00002.arrow"),
        #     # ])
        #     # # 처음 30개만 잘라내서 테스트
        #     # # train_table = train_table.slice(0, 30)
        #     # train_table = train_table.slice(0, 10)
            
        #     # # self.dataset = Dataset.from_dict(train_table.to_pydict())
        #     # self.dataset = Dataset.from_dict(train_table.to_pydict(), features=partial_features)
        #     # 오버 피팅 테스트
        #     validation_table = load_arrow_table(f"{dataset_dir}/cord-v2-validation.arrow")
        #     # 처음 10개만 잘라내서 테스트
        #     validation_table = validation_table.slice(0, 10)
        #     self.dataset = Dataset.from_dict(validation_table.to_pydict(), features=partial_features)
        #     # self.dataset = Dataset(pa_table=train_table)
        # elif self.split == "validation":
        #     validation_table = load_arrow_table(f"{dataset_dir}/cord-v2-validation.arrow")
        #     # 처음 10개만 잘라내서 테스트
        #     validation_table = validation_table.slice(0, 10)
        #     self.dataset = Dataset.from_dict(validation_table.to_pydict(), features=partial_features)
        # elif self.split == "test":
        #     self.dataset = Dataset.from_dict(load_arrow_table(f"{dataset_dir}/cord-v2-test.arrow").to_pydict(), features=partial_features)
        # else:
        #     raise ValueError(f"Invalid split: {self.split}")
        # self.dataset_length = len(self.dataset)
        
        from tqdm import tqdm
        from glob import glob
        import PIL
        
        self.gt_token_sequences = []
        # metadata_list = []
        dataset_dir = dataset_name_or_path
        with open(f"{dataset_dir}/metadata.jsonl", "r") as f:
            metadata_list = [json.loads(line) for line in f]
            if self.split == "train":
                metadata_list = metadata_list[:8]
            elif self.split == "validation":
                metadata_list = metadata_list[8:10]
            elif self.split == "test":
                pass
        self.dataset = Dataset.from_list(
            [
                {
                    'image' : PIL.Image.open(el.get('file_name')), 
                    'ground_truth' : json.dumps(el.get('ground_truth'))
                } 
                for el in metadata_list
            ]
        )
        self.dataset_length = len(self.dataset)
        pbar = tqdm(total=self.dataset_length, desc=f"Loading {self.split} dataset")
        # for sample in self.dataset:
        for sample in metadata_list:
            # if pbar.n>32:
            #     break
            # ground_truth = json.loads(sample["ground_truth"])
            ground_truth = sample["ground_truth"]
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            # dataset_dir_save = '/home/dasom/donut/dataset/cord_v2'
            # from pathlib import Path
            # from hashlib import md5
            # file_nm = md5(sample['image'].tobytes()).hexdigest()
            # Path(f'{dataset_dir_save}/{self.split}').mkdir(parents=True, exist_ok=True)
            # img_save_path = f"{dataset_dir_save}/{self.split}/{file_nm}.png"
            # if not os.path.exists(img_save_path):
            #     sample['image'].save(img_save_path)

            # sample2save = {
            #     'file_name': img_save_path,
            #     'ground_truth' : ground_truth
            # }

            # metadata_list.append(sample2save)
        # metadata_list
        # with open(f"{dataset_dir_save}/{self.split}/metadata.jsonl", "w", encoding="utf-8") as f:
        #     for item in metadata_list:
        #         f.write(json.dumps(item, ensure_ascii=False) + "\n")

            self.gt_token_sequences.append(
                [
                    task_start_token
                    + self.donut_model.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=self.sort_json_key,
                    )
                    + self.donut_model.decoder.tokenizer.eos_token
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )
            seq = self.gt_token_sequences[-1]
            # pbar.set_postfix({
            #     'number_of_token':len(self.donut_model.decoder.tokenizer(seq).get('input_ids')[0])
            # })
            print(seq)
            print('estimated number of tokens', len(self.donut_model.decoder.tokenizer(seq).get('input_ids')[0]))
            pbar.update(1)

        # { seq[0][:100] : len(self.donut_model.decoder.tokenizer(seq).get('input_ids')[0]) for seq in self.gt_token_sequences}
        self.donut_model.decoder.add_special_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]

        # input_tensor
        input_tensor = self.donut_model.encoder.prepare_input(sample["image"], random_padding=self.split == "train")

        # input_ids
        processed_parse = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse


class JSONParseEvaluator:
    """
    Calculate n-TED(Normalized Tree Edit Distance) based accuracy and F1 accuracy score
    """

    @staticmethod
    def flatten(data: dict):
        """
        Convert Dictionary into Non-nested Dictionary
        Example:
            input(dict)
                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }
            output(list)
                [
                    ("menu.name", "cake"),
                    ("menu.count", "2"),
                    ("menu.name", "juice"),
                    ("menu.count", "1"),
                ]
        """
        flatten_data = list()

        def _flatten(value, key=""):
            if type(value) is dict:
                for child_key, child_value in value.items():
                    _flatten(child_value, f"{key}.{child_key}" if key else child_key)
            elif type(value) is list:
                for value_item in value:
                    _flatten(value_item, key)
            else:
                flatten_data.append((key, value))

        _flatten(data)
        return flatten_data

    @staticmethod
    def update_cost(node1: Node, node2: Node):
        """
        Update cost for tree edit distance.
        If both are leaf node, calculate string edit distance between two labels (special token '<leaf>' will be ignored).
        If one of them is leaf node, cost is length of string in leaf node + 1.
        If neither are leaf node, cost is 0 if label1 is same with label2 othewise 1
        """
        label1 = node1.label
        label2 = node2.label
        label1_leaf = "<leaf>" in label1
        label2_leaf = "<leaf>" in label2
        if label1_leaf == True and label2_leaf == True:
            return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
        elif label1_leaf == False and label2_leaf == True:
            return 1 + len(label2.replace("<leaf>", ""))
        elif label1_leaf == True and label2_leaf == False:
            return 1 + len(label1.replace("<leaf>", ""))
        else:
            return int(label1 != label2)

    @staticmethod
    def insert_and_remove_cost(node: Node):
        """
        Insert and remove cost for tree edit distance.
        If leaf node, cost is length of label name.
        Otherwise, 1
        """
        label = node.label
        if "<leaf>" in label:
            return len(label.replace("<leaf>", ""))
        else:
            return 1

    def normalize_dict(self, data: Union[Dict, List, Any]):
        """
        Sort by value, while iterate over element if data is list
        """
        if not data:
            return {}

        if isinstance(data, dict):
            new_data = dict()
            for key in sorted(data.keys(), key=lambda k: (len(k), k)):
                value = self.normalize_dict(data[key])
                if value:
                    if not isinstance(value, list):
                        value = [value]
                    new_data[key] = value

        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                new_data = []
                for item in data:
                    item = self.normalize_dict(item)
                    if item:
                        new_data.append(item)
            else:
                new_data = [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
        else:
            new_data = [str(data).strip()]

        return new_data

    def cal_f1(self, preds: List[dict], answers: List[dict]):
        """
        Calculate global F1 accuracy score (field-level, micro-averaged) by counting all true positives, false negatives and false positives
        """
        total_tp, total_fn_or_fp = 0, 0
        for pred, answer in zip(preds, answers):
            pred, answer = self.flatten(self.normalize_dict(pred)), self.flatten(self.normalize_dict(answer))
            for field in pred:
                if field in answer:
                    total_tp += 1
                    answer.remove(field)
                else:
                    total_fn_or_fp += 1
            total_fn_or_fp += len(answer)
        return total_tp / (total_tp + total_fn_or_fp / 2)

    def construct_tree_from_dict(self, data: Union[Dict, List], node_name: str = None):
        """
        Convert Dictionary into Tree

        Example:
            input(dict)

                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }

            output(tree)
                                     <root>
                                       |
                                     menu
                                    /    \
                             <subtree>  <subtree>
                            /      |     |      \
                         name    count  name    count
                        /         |     |         \
                  <leaf>cake  <leaf>2  <leaf>juice  <leaf>1
         """
        if node_name is None:
            node_name = "<root>"

        node = Node(node_name)

        if isinstance(data, dict):
            for key, value in data.items():
                kid_node = self.construct_tree_from_dict(value, key)
                node.addkid(kid_node)
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                for item in data:
                    kid_node = self.construct_tree_from_dict(
                        item,
                        "<subtree>",
                    )
                    node.addkid(kid_node)
            else:
                for item in data:
                    node.addkid(Node(f"<leaf>{item}"))
        else:
            raise Exception(data, node_name)
        return node

    def cal_acc(self, pred: dict, answer: dict):
        """
        Calculate normalized tree edit distance(nTED) based accuracy.
        1) Construct tree from dict,
        2) Get tree distance with insert/remove/update cost,
        3) Divide distance with GT tree size (i.e., nTED),
        4) Calculate nTED based accuracy. (= max(1 - nTED, 0 ).
        """
        pred = self.construct_tree_from_dict(self.normalize_dict(pred))
        answer = self.construct_tree_from_dict(self.normalize_dict(answer))
        return max(
            0,
            1
            - (
                zss.distance(
                    pred,
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
                / zss.distance(
                    self.construct_tree_from_dict(self.normalize_dict({})),
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
            ),
        )

from .preprocessing import (UDDataset, LabelTokenizer, create_tokenizer,
                            create_label_tokenizer, PairSample)
from transformers import AutoTokenizer
from typing import List, Tuple
import torch
from torch import Tensor
from .config import bertje_name
from itertools import accumulate
from tabulate import tabulate


def grab_unpadded_labels(tokenss: List[List[int]], labels: List[int]):
    indices = [0]+list(accumulate(map(len, tokenss)))[:-1]
    return [labels[i] for i in indices]


def get_pred_list(preds: Tensor) -> List[int]:
    return torch.argmax(preds, dim=1).tolist()


def get_parse_result(postsent: str, tokens: List[int], prediction: Tensor,
                     tokenizer: AutoTokenizer, label_tokenizer: LabelTokenizer):
    tokenss = [tokenizer.tokenize(w) for w in postsent.replace('.', ' .').split()]
    pred_list = get_pred_list(prediction)
    start = tokens.index(2)
    end = start+len(tokens[start:])
    labels = grab_unpadded_labels(tokenss, pred_list[start:end][1:-1])
    return list(map(label_tokenizer.untokenize, labels))


def was_correct_parse(parse_result: Tuple[PairSample, List[str]]) -> bool:
    interpretation = parse_result[0].interpretation
    relpron_node = parse_result[1][2]
    body_noun_node = parse_result[1][4]
    if '{}' not in relpron_node or '{}' not in body_noun_node:
        return False
    else:
        relpron_label = relpron_node.split('{}')[1]
        body_noun_label = body_noun_node.split('{}')[1]
        if interpretation == 'subjrel':
            return relpron_label == 'nsubj' and body_noun_label == 'obj'
        elif interpretation == 'objrel':
            return relpron_label == 'obj' and body_noun_label == 'nsubj'


def calculate_accuracy(corrects: list[bool]):
    return round(100*(sum(corrects) / len(corrects)),2) if len(corrects) > 0 else None


def calculate_parse_accuracy(parse_results: List[Tuple[PairSample, List[str]]]):
    return calculate_accuracy(list(map(was_correct_parse, parse_results)))


def calculate_dict_accuracy_double_depth(results: dict[str, dict[str, List[Tuple[PairSample, List[str]]]]]):
    return {k: {k_inner: calculate_parse_accuracy(results[k][k_inner]) for k_inner in results[k]} for k in results}


def calculate_dict_accuracy_triple_depth(results: dict[str, dict[str, List[Tuple[PairSample, List[str]]]]]):
    return {k: {k_inner: {k_inner_inner: calculate_parse_accuracy(results[k][k_inner][k_inner_inner])
                          for k_inner_inner in results[k][k_inner]} for k_inner in results[k]} for k in results}


def filter_results_by_tags(results: List[Tuple[PairSample, List[str]]], data_tag: str, pre_tag: str, gen_tag: str):
    return [r for r in results if r[0].data_tag == data_tag and r[0].present_tag == pre_tag and r[0].postsent_tag == gen_tag]


def gather_by_category(results: List[Tuple[PairSample, List[str]]]):
    data_tags = ['irreversible', 'reversible-strong', 'reversible-weak']
    pre_tags = ['original', 'reversed']
    gen_tags = ['original', 'reversed']
    return {data_tag:
                {pre_tag:
                     {gen_tag: filter_results_by_tags(results, data_tag, pre_tag, gen_tag) for gen_tag in gen_tags}
                 for pre_tag in pre_tags}
            for data_tag in data_tags}


def gather_by_rev_outcome(results: List[Tuple[PairSample, List[str]]]):
    data_tags = ['irreversible', 'reversible-strong', 'reversible-weak']
    return {data_tag: {'subjrel': [r for r in results if r[0].data_tag == data_tag and r[0].interpretation == 'subjrel'],
                       'objrel': [r for r in results if r[0].data_tag == data_tag and r[0].interpretation == 'objrel']}
            for data_tag in data_tags}


def accuracy_by_category(results: List[Tuple[PairSample, List[str]]]):
    return calculate_dict_accuracy_triple_depth(gather_by_category(results))


def pprint_double_accs(accs):
    print("Accuracies by outcome:")
    for k in accs:
        print(k)
        for t1 in accs[k]:
            print(t1, '\t', accs[k][t1])


def pprint_triple_accs(accs):
    print("Accuracies by category:")
    for k in accs:
        print(k)
        for t1 in accs[k]:
            for t2 in accs[k][t1]:
                print(t1, '\t', t2, '\t', accs[k][t1][t2])


def calculate_dict_accuracy_single_depth(results: dict[str, List[Tuple[PairSample, List[str]]]]):
    return {k: calculate_parse_accuracy(results[k]) for k in results}


def filter_results_by_outcome(results: List[Tuple[PairSample, List[str]]]):
    return {'subjrel': [r for r in results if r[0].interpretation == 'subjrel'],
            'objrel': [r for r in results if r[0].interpretation == 'objrel']}


def accuracy_by_outcome(results: List[Tuple[PairSample, List[str]]]):
    return calculate_dict_accuracy_single_depth(filter_results_by_outcome(results))


def accuracy_by_rev_outcome(results: List[Tuple[PairSample, List[str]]]):
    return calculate_dict_accuracy_double_depth(gather_by_rev_outcome(results))


def analysis_no_context(dataset: UDDataset, predictions: List[Tensor]):
    tokenizer = create_tokenizer(bertje_name)
    label_tokenizer = create_label_tokenizer()
    parse_results = [(d.sample, get_parse_result(d.sample.postsent, d.sentence_tokens, pred, tokenizer, label_tokenizer))
               for d, pred in zip(dataset, predictions)]
    return accuracy_by_rev_outcome(parse_results)


def analysis_context(dataset: UDDataset, predictions: List[Tensor]):
    tokenizer = create_tokenizer(bertje_name)
    label_tokenizer = create_label_tokenizer()
    parse_results = [(d.sample, get_parse_result(d.sample.postsent, d.sentence_tokens, pred, tokenizer, label_tokenizer))
               for d, pred in zip(dataset, predictions)]
    return accuracy_by_category(parse_results)


def print_as_latex_table_data(accs):
    data = []
    for k in accs:
        data.append([f"\\textbf{{{k.capitalize()}}}", accs[k]['original']['original'], accs[k]['original']['reversed'],
                        accs[k]['reversed']['original'], accs[k]['reversed']['reversed']])
    print(tabulate(data, tablefmt="latex_raw"))
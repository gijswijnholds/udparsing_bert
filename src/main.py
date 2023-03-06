from transformers import AutoModelForTokenClassification
from .preprocessing import prepare_datasets, prepare_finetune_datasets, tokenize_sequence_present
from .trainer import Trainer
from .analysis import analysis_no_context, analysis_context, pprint_double_accs, pprint_triple_accs, print_as_latex_table_data
from .config import bertje_name, X, finetune_settings
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import os

def load_model(path: str = None):
    if path:
        return AutoModelForTokenClassification.from_pretrained(path, num_labels=X)
    else:
        return AutoModelForTokenClassification.from_pretrained(bertje_name, num_labels=X)


def setup_trainer(bert_name: str,
                  device: str,
                  results_folder: str="./results",
                  model_folder: str="./models",
                  train_batch_size: int = 32,
                  tokenize_present: bool = False) -> Trainer:
    model = load_model()
    if tokenize_present:
        train_dataset, val_dataset, test_dataset = prepare_datasets(bert_name, tokenize_fn=tokenize_sequence_present)
    else:
        train_dataset, val_dataset, test_dataset = prepare_datasets(bert_name)
    return Trainer(name=f'{bert_name.split("/")[-1]}_udparsing_{tokenize_present}',
                   model=model,
                   train_dataset=train_dataset,
                   val_dataset=val_dataset,
                   test_dataset=test_dataset,
                   batch_size_train=train_batch_size,
                   batch_size_val=128,
                   batch_size_test=128,
                   optim_constructor=AdamW,
                   lr=1e-04,
                   device=device,
                   word_pad=3,
                   results_folder=results_folder,
                   model_folder=model_folder)


def setup_finetuner(bert_name: str,
                    model_path: str,
                    device: str,
                    data_path: str,
                    results_folder: str="./results",
                    model_folder: str="./models",
                    train_batch_size: int = 32,
                    contextualize: bool = False,
                    parse_present: bool = False,
                    freeze_lm: bool = False) -> Trainer:
    model = load_model(model_path)
    train_dataset, val_dataset, test_dataset = prepare_finetune_datasets(data_path,
                                                                         contextualize=contextualize,
                                                                         parse_present=parse_present)
    setting_name = data_path.split('/')[-1].split('samples_')[-1]
    return Trainer(name=model_path.split('/')[-1]+'_'+setting_name+f'_frozen_{freeze_lm}',
                   model=model,
                   train_dataset=train_dataset,
                   val_dataset=val_dataset,
                   test_dataset=test_dataset,
                   batch_size_train=train_batch_size,
                   batch_size_val=128,
                   batch_size_test=128,
                   optim_constructor=AdamW,
                   lr=1e-04,
                   device=device,
                   word_pad=3,
                   results_folder=results_folder,
                   model_folder=model_folder,
                   freeze_lm=freeze_lm)

def my_main():
    trainer = setup_trainer(bertje_name, device='cpu', train_batch_size=32, tokenize_present=True)
    trainer.train_loop(num_epochs=20, val_every=1, save_at_best=True)


def finetune_all_params():
    for setting in finetune_settings:
        parse_present = True if 'svo' in setting else False
        print(f"Finetuning a full model for setting {setting} now...")
        print(f"Parsing present: {parse_present}")
        trainer = setup_finetuner(bertje_name, model_path='./models/bert-base-dutch-cased_udparsing_True_10',
                                  device='cpu', data_path=f'./export_data/{setting}', train_batch_size=32,
                                  contextualize=True, parse_present=parse_present, freeze_lm=False)
        trainer.train_loop(num_epochs=5, val_every=1, save_at_best=True)
        print(f"Finetuning a frozen LM model for setting {setting} now...")
        trainer_frozen = setup_finetuner(bertje_name, model_path='./models/bert-base-dutch-cased_udparsing_True_10',
                                  device='cpu', data_path=f'./export_data/{setting}', train_batch_size=32,
                                  contextualize=True, parse_present=parse_present, freeze_lm=True)
        trainer_frozen.train_loop(num_epochs=5, val_every=1, save_at_best=True)
        print(f"Done finetuning for setting {setting}!")


def test_finetuner(setting: str, freeze_lm: bool=False, parse_present: bool=False):
    setting_name = setting.split('samples_')[-1]
    model_path_base = f'bert-base-dutch-cased_udparsing_True_10_{setting_name}_frozen_{freeze_lm}'
    model_path = f"./models/{[fn for fn in os.listdir('./models') if fn.startswith(model_path_base)][0]}"
    print(f"Testing for setting {setting}...")
    print(f"The Language Model is frozen: {freeze_lm}")
    no_context_model = setup_finetuner(bertje_name, model_path=model_path, device='cpu',
                                       data_path=f'./export_data/{setting}', train_batch_size=32,
                                       contextualize=False, parse_present=parse_present, freeze_lm=freeze_lm)
    no_context_results = analysis_no_context(no_context_model.test_loader.dataset, no_context_model.predict_epoch())
    context_model = setup_finetuner(bertje_name, model_path=model_path, device='cpu',
                                    data_path=f'./export_data/{setting}', train_batch_size=32,
                                    contextualize=True, parse_present=parse_present, freeze_lm=freeze_lm)
    context_results = analysis_context(context_model.test_loader.dataset, context_model.predict_epoch())
    return no_context_results, context_results


def test_all_finetuners():
    all_results = {}
    cnt = 0
    for setting in finetune_settings:
        cnt += 1
        print(f"Testing model #{cnt}...")
        parse_present = True if 'svo' in setting else False
        no_context_results, context_results = test_finetuner(setting, freeze_lm=False, parse_present=parse_present)
        no_context_results_frozen, context_results_frozen = test_finetuner(setting, freeze_lm=True, parse_present=parse_present)
        all_results[setting] = {'no_context': no_context_results, 'context': context_results,
                                'no_context_frozen': no_context_results_frozen, 'context_frozen': context_results_frozen}
    return all_results


def print_results(results):
    print("---------------------RESULTS---------------------")
    for setting in results:
        print(setting)
        # print("No context:")
        # pprint_double_accs(results[setting]['no_context'])
        print("Context:")
        print_as_latex_table_data(results[setting]['context'])
        # print("No context (Frozen LM):")
        # pprint_double_accs(results[setting]['no_context_frozen'])
        print("Context (Frozen LM):")
        print_as_latex_table_data(results[setting]['context_frozen'])
    print("---------------------END OF RESULTS---------------------")

#     Test set eval_epoch (accuracy)
# (0.7958968877792358, 0.8691629998789174)
#     Test set UAS/LAS (LAS == accuracy)
# UAS: 0.8836690272650809, LAS: 0.8691629998789174
import os
import shutil
import pickle
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch import LongTensor, Tensor, no_grad
from typing import Callable, Any, List, Tuple, Dict
from typing import Optional as Maybe
from .preprocessing import Sample, UDDataset, LabelTokenizer
from transformers import AutoModelForTokenClassification


def sequence_collator(word_pad: int) -> Callable[[List[Sample]], Tuple[Tensor, Tensor, Tensor]]:
    def collate_fn(samples: List[Sample]) -> Tuple[Tensor, Tensor, Tensor]:
        input_ids = pad_sequence([torch.tensor(sample.sentence_tokens) for sample in samples],
                                 padding_value=word_pad, batch_first=True)
        try:
            input_mask = pad_sequence([torch.tensor(sample.sentence_mask) for sample in samples],
                                           padding_value=0, batch_first=True)
        except AttributeError:
            input_mask = input_ids != word_pad
        labels = pad_sequence([torch.tensor(sample.labels_tokens) for sample in samples],
                                 padding_value=-100, batch_first=True)
        return input_ids, input_mask, labels
    return collate_fn


def compute_uas_las(predictions: Tensor, trues: Tensor, label_tokenizer: LabelTokenizer) -> Tuple[float, float]:
    valid_trues = trues != -100
    predicted_str = list(map(label_tokenizer.untokenize, torch.argmax(predictions, dim=2)[valid_trues].tolist()))
    true_str = list(map(label_tokenizer.untokenize, trues[valid_trues].tolist()))
    uas = sum([p.split('{}')[0] == t.split('{}')[0] for (p, t) in zip(predicted_str, true_str)]) / len(predicted_str)
    las = sum([p == t for (p, t) in zip(predicted_str, true_str)]) / len(predicted_str)
    return uas, las


# TODO: acc UAS/LAS accuracy for analysis later
def compute_accuracy(predictions: Tensor, trues: Tensor) -> float:
    valid_trues = trues != -100
    return (trues[valid_trues] == torch.argmax(predictions, dim=2)[valid_trues]).float().mean().item()


def compute_accuracy_mps(predictions: torch.Tensor, trues: torch.Tensor) -> float:
    _, predicted_classes = torch.max(predictions, dim=2)

    accuracy = 0.0
    count = 0
    for i in range(trues.size(0)):
        for j in range(trues.size(1)):
            if trues[i][j] == -100:
                continue
            count += 1
            if trues[i][j] == predicted_classes[i][j]:
                accuracy += 1
    return accuracy / count


class Trainer:
    def __init__(self,
                 name: str,
                 model: AutoModelForTokenClassification,
                 word_pad: int,
                 train_dataset: Maybe[UDDataset] = None,
                 val_dataset: Maybe[UDDataset] = None,
                 test_dataset: Maybe[UDDataset] = None,
                 batch_size_train: Maybe[int] = None,
                 batch_size_val: Maybe[int] = None,
                 batch_size_test: Maybe[int] = None,
                 optim_constructor: Maybe[type] = None,
                 lr: Maybe[float] = None,
                 device: str = 'cuda',
                 results_folder: str = './results',
                 model_folder: str = './models',
                 freeze_lm: bool = False):
        self.name = name
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.device = device
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,
                                       collate_fn=sequence_collator(word_pad)) if train_dataset else None
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False,
                                     collate_fn=sequence_collator(word_pad)) if val_dataset else None
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False,
                                      collate_fn=sequence_collator(word_pad)) if test_dataset else None
        if freeze_lm:
            for param in model.bert.parameters():
                param.requires_grad = False
            self.model = model.to(device)
            self.optimizer = optim_constructor(self.model.classifier.parameters(), lr=lr) if optim_constructor else None
        else:
            self.model = model.to(device)
            self.optimizer = optim_constructor(self.model.parameters(), lr=lr) if optim_constructor else None
        self.results_folder = results_folder
        self.model_folder = model_folder

    def save_results(self, results: Dict[int, Dict[str, float]]):
        file_path = f"{self.results_folder}/results_{self.name}.p"
        if os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path, 'wb') as outf:
            pickle.dump(results, outf)

    def train_batch(
            self,
            batch: Tuple[LongTensor, LongTensor, LongTensor]) -> Tuple[float, float]:
        self.model.train()
        input_ids, input_masks, ys = batch
        outputs = self.model.forward(input_ids.to(self.device), input_masks.to(self.device), labels=ys.to(self.device))
        batch_loss = outputs.loss
        accuracy = compute_accuracy_mps(outputs.logits, ys.to(self.device))
        batch_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return batch_loss.item(), accuracy

    def train_epoch(self):
        epoch_loss, epoch_accuracy = 0., 0.
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                loss, accuracy = self.train_batch(batch)
                tepoch.set_postfix(loss=loss, accuracy=accuracy)
                epoch_loss += loss
                epoch_accuracy += accuracy
        return epoch_loss / len(self.train_loader), epoch_accuracy / len(self.train_loader)

    @no_grad()
    def eval_batch(
            self,
            batch: Tuple[LongTensor, LongTensor, LongTensor]) -> Tuple[float, float]:
        self.model.eval()
        input_ids, input_masks, ys = batch
        outputs = self.model.forward(input_ids.to(self.device), input_masks.to(self.device), labels=ys.to(self.device))
        batch_loss = outputs.loss
        accuracy = compute_accuracy_mps(outputs.logits, ys.to(self.device))
        return batch_loss.item(), accuracy

    def eval_epoch(self, eval_set: str):
        epoch_loss, epoch_accuracy = 0., 0.
        loader = self.val_loader if eval_set == 'val' else self.test_loader
        batch_counter = 0
        with tqdm(loader, unit="batch") as tepoch:
            for batch in tepoch:
                batch_counter += 1
                loss, accuracy = self.eval_batch(batch)
                tepoch.set_postfix(loss=loss, accuracy=accuracy)
                epoch_loss += loss
                epoch_accuracy += accuracy
        return epoch_loss / len(loader), epoch_accuracy / len(loader)

    @no_grad()
    def predict_batch(
            self,
            batch: Tuple[LongTensor, LongTensor, Any]) -> List[int]:
        self.model.eval()
        input_ids, input_masks, _ = batch
        predictions = self.model.forward(input_ids.to(self.device), input_masks.to(self.device)).logits
        return predictions

    @no_grad()
    def predict_epoch(self) -> List[int]:
        return [label for batch in tqdm(self.test_loader) for label in self.predict_batch(batch)]

    def train_loop(self, num_epochs: int, val_every: int = 1, save_at_best: bool = False):
        results = dict()
        for e in range(num_epochs):
            print(f"Epoch {e}...")
            train_loss, train_acc = self.train_epoch()
            print(f"Train loss {train_loss:.5f}, Train accuracy: {train_acc:.5f}")
            if (e % val_every == 0 and e != 0) or e == num_epochs - 1:
                val_loss, val_acc = self.eval_epoch(eval_set='val')
                print(f"Val loss {val_loss:.5f}, Val accuracy: {val_acc:.5f}")
                if save_at_best and val_acc > max([v['val_acc'] for v in results.values()]):
                    for folder in os.listdir(self.model_folder):
                        if folder.startswith(f'{self.name}'):
                            shutil.rmtree(os.path.join(self.model_folder, folder))
                    self.model.save_pretrained(f'{self.model_folder}/{self.name}_{e}')
            else:
                val_loss, val_acc = None, -1
            results[e] = {'train_loss': train_loss, 'train_acc': train_acc,
                          'val_loss': val_loss, 'val_acc': val_acc}
            self.save_results(results)
        print(f"Best epoch was {max(results, key=lambda k: results[k]['val_acc'])}")
        return results
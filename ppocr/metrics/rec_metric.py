# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import Levenshtein
import string
from nltk.metrics.distance import edit_distance


def string_cleaner(s):
    return (s
        .replace("“", "\"")
        .replace("”", "\"")
        .replace("''", "\"")
        .replace("‘‘", "\"")
        .replace("’’", "\"")
        .replace("\n", "")
    )


def textline_evaluation(
        pairs,
        print_incorrect=False, 
        no_spaces_in_eval=False, 
        norm_edit_distance=False, 
        uncased=False
    ):

    n_correct = 0
    edit_count = 0
    length_of_data = len(pairs)
    n_chars = sum(len(gt) for gt, _ in pairs)

    for gt, pred in pairs:

        # eval w/o spaces
        pred, gt = string_cleaner(pred), string_cleaner(gt)
        gt = gt.strip() if not no_spaces_in_eval else gt.strip().replace(" ", "")
        pred = pred.strip() if not no_spaces_in_eval else pred.strip().replace(" ", "")
        if uncased:
            pred, gt = pred.lower(), gt.lower()
        
        # textline accuracy
        if pred == gt:
            n_correct += 1
        else:
            if print_incorrect:
                print(f"GT: {gt}\nPR: {pred}\n")

        # ICDAR2019 Normalized Edit Distance
        if norm_edit_distance:
            if len(gt) > len(pred):
                edit_count += edit_distance(pred, gt) / len(gt)
            else:
                edit_count += edit_distance(pred, gt) / len(pred)
        else:
            edit_count += edit_distance(pred, gt)

    accuracy = n_correct / float(length_of_data) * 100
    
    if norm_edit_distance:
        cer = edit_count / float(length_of_data)
    else:
        cer = edit_count / n_chars

    return accuracy, cer


class RecMetric(object):
    def __init__(self,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=True,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        self.preds = preds
        self.labels = labels
        correct_num = 0
        all_num = 0
        char_num = 0
        norm_edit_dis = 0.0
        cer = 0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            norm_edit_dis += Levenshtein.distance(pred, target) / max(
                len(pred), len(target), 1)
            cer += edit_distance(pred, target)
            assert edit_distance(pred, target) == Levenshtein.distance(pred, target)
            if pred == target:
                correct_num += 1
            all_num += 1
            char_num += len(target)
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        self.cer += cer
        self.char_num += char_num
        return {
            'acc': correct_num / (all_num + self.eps),
            'norm_edit_dis': 1 - norm_edit_dis / (all_num + self.eps),
            '1-cer': 1 - (cer / char_num),
            'cer': cer / char_num,
            'cer_custom': textline_evaluation(
                [(target, pred) for (pred, _), (target, _) in zip(preds, labels)],
                print_incorrect=False, 
                no_spaces_in_eval=False, 
                norm_edit_distance=False, 
                uncased=True
            )
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        cer = self.cer / self.char_num
        cer_custom = textline_evaluation(
            [(target, pred) for (pred, _), (target, _) in zip(self.preds, self.labels)],
            print_incorrect=False, 
            no_spaces_in_eval=False, 
            norm_edit_distance=False, 
            uncased=False
        )
        self.reset()
        return {'acc': acc, 'norm_edit_dis': norm_edit_dis, 'cer': cer, '1-cer': 1 - cer, 'cer_custom': cer_custom}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
        self.cer = 0
        self.char_num = 0

import os
import json
import time
import random
import logging
import unicodedata
from typing import Any, Callable, Optional

import torch
import numpy as np
import torch.nn as nn
from peft.utils import WEIGHTS_NAME
from huggingface_hub import hf_hub_download
from peft import PeftModel, LoraConfig, set_peft_model_state_dict


def load_json(input_file):
    with open(input_file, "r") as f:
        samples = json.load(f)
    return samples


def load_dict(dict_path):
    """load_dict"""
    vocab = {}
    for line in open(dict_path, "r", encoding="utf-8"):
        key, value = line.strip("\n").split("\t")
        vocab[int(key)] = value
    return vocab


def write_dict(dict_path, dict_data):
    with open(dict_path, "w", encoding="utf-8") as f:
        for key, value in dict_data.items():
            f.writelines("{}\t{}\n".format(key, value))


def str_q2b(text):
    ustring = text
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


class ProgressBar(object):
    """
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='Training')
        >>> step = 2
        >>> pbar(step=step)
    """

    def __init__(self, n_total, width=30, desc="Training"):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f"[{self.desc}] {current}/{self.n_total} ["
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += "=" * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += "="
        bar += "." * (self.width - prog_width)
        bar += "]"
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = "%d:%02d:%02d" % (
                    eta // 3600,
                    (eta % 3600) // 60,
                    eta % 60,
                )
            elif eta > 60:
                eta_format = "%d:%02d" % (eta // 60, eta % 60)
            else:
                eta_format = "%ds" % eta
            time_info = f" - ETA: {eta_format}"
        else:
            if time_per_unit >= 1:
                time_info = f" {time_per_unit:.1f}s/step"
            elif time_per_unit >= 1e-3:
                time_info = f" {time_per_unit * 1e3:.1f}ms/step"
            else:
                time_info = f" {time_per_unit * 1e6:.1f}us/step"

        show_bar += time_info
        if len(info) != 0:
            show_info = f"{show_bar} " + "-".join(
                [f" {key}: {value:.4f} " for key, value in info.items()]
            )
            print(show_info, end="")
        else:
            print(show_bar, end="")


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != "":
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)
    return logger


class TokenRematch:
    def __init__(self):
        self._do_lower_case = True

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）"""
        if token[:2] == "##":
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_control(ch):
        """控制类字符判断"""
        return unicodedata.category(ch) in ("Cc", "Cf")

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号"""
        return bool(ch) and (ch[0] == "[") and (ch[-1] == "]")

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系"""
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = "", []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize("NFD", ch)
                ch = "".join([c for c in ch if unicodedata.category(c) != "Mn"])
            ch = "".join(
                [
                    c
                    for c in ch
                    if not (
                        ord(c) == 0 or ord(c) == 0xFFFD or self._is_control(c)
                    )
                ]
            )
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping


class LoRAModel:
    def __init__(self, base_model_factory: Callable[..., Any]) -> None:
        self.base_model_factory = base_model_factory

    def get_base_model(self, *args: Any, **kwargs: Any) -> nn.Module:
        return self.base_model_factory(*args, **kwargs)

    def get_model_state_dict(
        self, base_model: nn.Module, loaded_state_dict: dict
    ) -> dict:
        prefix = getattr(base_model, "base_model_prefix", None)
        if not prefix:
            return loaded_state_dict

        peft_prefix = "base_model.model."
        loaded_keys = {
            s[len(peft_prefix) :]: tensor
            for s, tensor in loaded_state_dict.items()
            if s.startswith(peft_prefix)
        }
        loaded_keys_not_prefixed = {
            s: tensor
            for s, tensor in loaded_state_dict.items()
            if not s.startswith(peft_prefix)
        }

        model_state_dict = base_model.state_dict()
        expected_keys = list(model_state_dict.keys())
        has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
        expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)

        remove_prefix_from_model = (
            has_prefix_module and not expects_prefix_module
        )
        add_prefix_to_model = not has_prefix_module and expects_prefix_module

        _prefix = f"{prefix}."
        if remove_prefix_from_model:
            prefix_removed = {
                (s[len(_prefix) :] if s.startswith(_prefix) else s): tensor
                for s, tensor in loaded_keys.items()
            }
            return {**loaded_keys_not_prefixed, **prefix_removed}
        elif add_prefix_to_model:
            prefix_added = {
                (_prefix + s if not s.startswith(_prefix) else s): tensor
                for s, tensor in loaded_keys.items()
            }
            return {**loaded_keys_not_prefixed, **prefix_added}
        return loaded_state_dict

    def from_pretrained(self, model_id: str, *args: Any, **kwargs: Any):
        config = LoraConfig.from_pretrained(model_id)
        base_model = self.get_base_model(*args, **kwargs)
        model = PeftModel(base_model, config)

        if os.path.exists(os.path.join(model_id, WEIGHTS_NAME)):
            filename = os.path.join(model_id, WEIGHTS_NAME)
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME)
            except:  # noqa
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} is present at {model_id}."
                )

        adapters_weights = torch.load(
            filename,  # type: ignore
            map_location=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )
        set_peft_model_state_dict(
            model, self.get_model_state_dict(base_model, adapters_weights)
        )
        return model

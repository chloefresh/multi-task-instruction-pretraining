import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import math
from typing import List
from dataclasses import dataclass, field
from typing import Optional
from datasets import disable_caching
disable_caching()

import logging
import json
import torch
from transformers.utils import add_start_docstrings
import transformers
from datasets import load_dataset,concatenate_datasets
import copy
import glob
from itertools import chain
from glob import glob

from utils.prompter import Prompter

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import (
    AutoConfig,
    BloomForCausalLM,
    AutoModelForCausalLM,
    AutoModel,
    LlamaTokenizer,
    LlamaForCausalLM,
    BloomTokenizerFast,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import get_last_checkpoint
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.utils import add_start_docstrings
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100

MODEL_CLASSES = {
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoConfig, AutoModel, AutoTokenizer),
    "llama": (AutoConfig, LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_type: str = field(
        default=None,
        metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    llama: bool = field(
        default=False,
        metadata={"help": "Llama model"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})

    def __post_init__(self):
        if self.model_type is None:
            raise ValueError("You must specify a valid model_type to run training. Available model types are " + ", ".join(
                    MODEL_CLASSES.keys()))
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")



@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    prompt_template: Optional[str] = field(
        default='qa', metadata={"help": "The type of template file."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    pt_train_file_dir: Optional[str] = field(default=None, metadata={"help": "The train text data file folder."})
    pt_validation_file_dir: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on text file folder."},
    )
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class TrainingArguments(TrainingArguments):
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."},
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA."}
    )
    use_int8_training: bool = field(
        default=False,
        metadata={"help": "Whether to use int8 training."}
    )
    lora_config: Optional[str] = field(
        default=None,
        metadata={"help": "LoRA config file."},
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "ddp_find_unused_parameters"}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "gradient_checkpointing"}
    )

# save peft at train end
class SavePeftModelAtEndCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        return control


def print_rank_0(msg, log_file, rank=0):
    if rank <= 0:
        with open(log_file, 'a') as f:
            print(msg)
            f.write(msg + '\n')

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    global_rank = torch.distributed.get_rank()
    log_file = os.path.join(training_args.output_dir,'print_log.txt')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    prompter = Prompter(data_args.prompt_template)
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    if not model_args.model_type:
        raise ValueError("Please specify a model_type, e.g. llama, chatglm, bloom, etc.")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    #pretraining datasets tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 2048:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 2048. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    data_files = {}
    dataset_args = {}
    logger.info(f"pt_train files_dir: {data_args.pt_train_file_dir}")
    if data_args.pt_train_file_dir is not None and os.path.exists(data_args.pt_train_file_dir):
        train_data_files = glob(f'{data_args.pt_train_file_dir}/**/*.txt', recursive=True) + glob(
            f'{data_args.pt_train_file_dir}/**/*.json', recursive=True) + glob(
            f'{data_args.pt_train_file_dir}/**/*.jsonl', recursive=True)
        logger.info(f"train files: {train_data_files}")
        # Train data files must be same type, e.g. all txt or all jsonl
        types = [f.split('.')[-1] for f in train_data_files]
        if len(set(types)) > 1:
            raise ValueError(f"train files must be same type, e.g. all txt or all jsonl, but got {types}")
        data_files["train"] = train_data_files
    if data_args.pt_validation_file_dir is not None and os.path.exists(data_args.pt_validation_file_dir):
        eval_data_files = glob(f'{data_args.pt_validation_file_dir}/**/*.txt', recursive=True) + glob(
            f'{data_args.pt_validation_file_dir}/**/*.json', recursive=True) + glob(
            f'{data_args.pt_validation_file_dir}/**/*.jsonl', recursive=True)
        logger.info(f"eval files: {eval_data_files}")
        data_files["validation"] = eval_data_files
        # Train data files must be same type, e.g. all txt or all jsonl
        types = [f.split('.')[-1] for f in eval_data_files]
        if len(set(types)) > 1:
            raise ValueError(f"train files must be same type, e.g. all txt or all jsonl, but got {types}")
    extension = "text" if data_files["train"][0].endswith('txt') else 'json'
    if extension == "text":
        dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        **dataset_args,
    )

    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
            **dataset_args,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
            **dataset_args,
        )
    
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    with training_args.main_process_first(desc="Dataset tokenization and grouping"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )
        
    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets['train']
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.debug(f"Num train_samples: {len(train_dataset)}")
        logger.debug("Tokenized training example:")
        logger.debug(tokenizer.decode(train_dataset[0]['input_ids']))

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.debug(f"Num eval_samples: {len(eval_dataset)}")
        logger.debug("Tokenized eval example:")
        logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))
    
    if model_args.model_type and model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        config = config_class.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir
        )
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            load_in_8bit=model_args.load_in_8bit,
            # device_map=model_args.device_map,
            trust_remote_code=model_args.trust_remote_code,
        )




    # int8 is not compatible with DeepSpeed (require not to pass device_map)
    if training_args.use_int8_training:
        print_rank_0("int8 is not compatible with DeepSpeed. ", log_file, global_rank)
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if world_size != 1 else "auto"
        # device_map = "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=True,      # xxx: int8 load in
            device_map=device_map,  # xxx: int8 requires passing device_map
            torch_dtype=torch_dtype,
        )

    if model_args.llama:
        print_rank_0("Set the eos_token_id and bos_token_id of LLama model tokenizer", log_file, global_rank)
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"  # Allow batched inference
    else:
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"  # Allow batched inference

    print_rank_0("tokenizer.eos_token_id = {}".format(tokenizer.eos_token_id), log_file, global_rank)
    print_rank_0("tokenizer.pad_token_id = {}".format(tokenizer.pad_token_id), log_file, global_rank)
    print_rank_0("tokenizer.bos_token_id = {}".format(tokenizer.bos_token_id), log_file, global_rank)

    # peft model
    if training_args.use_lora:
        print_rank_0("Loading lora config from {}".format(training_args.lora_config), log_file, global_rank)
        lora_config = json.load(open(training_args.lora_config))
        print_rank_0("Lora config: {}".format(lora_config), log_file, global_rank)
        if training_args.use_int8_training:
            print_rank_0("training_args.use_int8_training!!! (int8 is not compatible with DeepSpeed)", log_file, global_rank)
            model = prepare_model_for_int8_training(model)
        config = LoraConfig(
            r=lora_config['lora_r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=lora_config['lora_target_modules'],
            lora_dropout=lora_config['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = get_peft_model(model, config)

        # In LoRA mode, add "--resume_from_checkpoint saved_models_lora/checkpoint-xxx" in run.sh to resume training
        # If resuming training, better not change your hyper-parameters
        if training_args.resume_from_checkpoint:
            # Check the available weights and load them
            checkpoint_name = os.path.join(
                training_args.resume_from_checkpoint, "pytorch_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            # The two files above have a different name depending on how they were saved, but are actually the same.
            if os.path.exists(checkpoint_name):
                print(f"In LoRA mode, restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"In LoRA mode, checkpoint {checkpoint_name} not found, training from scratch")
    
    model = model.float()
    print_trainable_parameters(model) 

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # model.is_parallelizable = True
    # model.model_parallel = True

    def generate_and_tokenize_prompt(data_point):
        input_ids = []
        labels = []
        sentence_value = prompter.generate_prompt(data_point["instruction"].lower())
        sentence_ids = tokenizer.encode(sentence_value, add_special_tokens=False)#do not add bos_token_id
        input_ids += sentence_ids
        labels += [IGNORE_INDEX] * len(sentence_ids)
        # add eos at every end of assistant sentence
        sentence_value = data_point["output"].lower()
        sentence_ids = tokenizer.encode(sentence_value, add_special_tokens=False)#do not add bos_token_id
        label = copy.deepcopy(sentence_ids)
        input_ids += sentence_ids
        labels += label

        # add eos at every end of assistant sentence
        input_ids += [tokenizer.eos_token_id]#make sure eos_token_id is correct
        labels += [tokenizer.eos_token_id]

        input_ids = input_ids[:training_args.model_max_length-1]
        labels = labels[:training_args.model_max_length-1]
        if not any(x > -100 for x in labels):
            labels[18:24] = input_ids[18:24]#labels can not have all values being -100. 18 and 24 are just random numbers

        attention_mask = [1] * len(input_ids)
        tokenized_full_prompt = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        return tokenized_full_prompt

    assert os.path.exists(data_args.train_file), "{} file not exists".format(data_args.train_file)
    if data_args.train_file.endswith(".json") or data_args.train_file.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_args.train_file, cache_dir=model_args.cache_dir)
    else:
        data = load_dataset(data_args.train_file, cache_dir=model_args.cache_dir)

    data.cleanup_cache_files()
    column_names = list(data["train"].features)
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt,remove_columns=column_names)
    val_data = load_dataset("json", data_files=data_args.validation_file, cache_dir=model_args.cache_dir)
    val_data = val_data["train"].shuffle().map(generate_and_tokenize_prompt,remove_columns=column_names)

    for i in range(2):
        print_rank_0("Eval tokenized example: {}".format(val_data[i]), log_file, global_rank)
    for i in range(2):
        print_rank_0("Train tokenized example: {}".format(train_data[i]), log_file, global_rank)

    training_nums = len(data['train'])
    num_gpus = torch.cuda.device_count()


    batch_size = training_args.per_device_train_batch_size * training_args.world_size * training_args.gradient_accumulation_steps
    t_total = math.ceil(training_nums/batch_size) * training_args.num_train_epochs
    training_args.eval_steps = max(t_total // 2, 5)
    training_args.save_steps = training_args.eval_steps
    training_args.warmup_steps = int(t_total*training_args.warmup_ratio) if training_args.warmup_ratio>0.0 else training_args.warmup_steps
    print_rank_0("num_gpus = {}, training_nums = {}, t_total = {}, warmup_steps = {}, eval_steps = {}, save_steps = {}".format(num_gpus, training_nums, t_total, training_args.warmup_steps, training_args.eval_steps, training_args.save_steps), log_file, global_rank)
    print_rank_0("val data nums = {}, training_nums = {}, batch_size = {}".format(len(val_data), training_nums, batch_size), log_file, global_rank)


    merged_train_dataset = concatenate_datasets([train_dataset,train_data])
    #Trainer
    #https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
    #https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py
    #https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
    #https://www.deepspeed.ai/docs/config-json/
    #https://huggingface.co/docs/accelerate/usage_guides/deepspeed
    #https://huggingface.co/transformers/v4.10.1/main_classes/deepspeed.html
    #https://github.com/tatsu-lab/stanford_alpaca/issues/176
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=merged_train_dataset,
        eval_dataset=val_data,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )
    print_rank_0(f"Using {training_args.half_precision_backend} half precision backend", log_file, global_rank)
    # Train!
    len_dataloader = len(trainer.get_train_dataloader())
    num_update_steps_per_epoch = len_dataloader // training_args.gradient_accumulation_steps

    total_train_batch_size = training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    num_examples = trainer.num_examples(trainer.get_train_dataloader())
    num_train_samples = num_examples * training_args.num_train_epochs
    max_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)

    print_rank_0("***** Running training *****", log_file, global_rank)
    print_rank_0(f"  Num examples = {num_examples}", log_file, global_rank)
    print_rank_0(f"  Num train samples = {num_train_samples}", log_file, global_rank)
    print_rank_0(f"  world_size = {world_size}", log_file, global_rank)
    print_rank_0(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}", log_file, global_rank)
    print_rank_0(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}", log_file, global_rank)
    print_rank_0(f"  Total optimization steps = {max_steps}", log_file, global_rank)
#    print_rank_0(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True)}", log_file, global_rank)
    
    model.config.use_cache = False
    if training_args.use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))

    if training_args.use_lora:    # lora training
        trainer.train(resume_from_checkpoint=None)
    else:    # full-parameter tuning
        if training_args.resume_from_checkpoint:    # resume training
            checkpoint_name = os.path.join(
                training_args.resume_from_checkpoint, "pytorch_model.bin"
            )
            if os.path.exists(checkpoint_name):
                print(f"In full-params mode, restarting from {checkpoint_name}...")
                trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            else:
                print(f"In full-params mode, checkpoint {checkpoint_name} not found, training from scratch...")
                trainer.train(resume_from_checkpoint=None)
        else:    # training from scratch
            trainer.train(resume_from_checkpoint=None)
    if training_args.use_lora:
        model.save_pretrained(training_args.output_dir)#Save adapter_model.bin and adapter_config.json
    
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        logger.debug(f"Eval metrics: {metrics}")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    trainer.save_model() # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2808
    print_rank_0("\n Training completed!!! If there's a warning about missing keys above, please disregard :)", log_file, global_rank)


if __name__ == "__main__":
    main()

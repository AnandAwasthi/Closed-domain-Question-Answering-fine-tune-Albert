# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, XLNetConfig, XLMConfig)),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
}


def set_seed(self):
    random.seed(self.seed)
    np.random.seed(self.seed)
    torch.manual_seed(self.seed)
    if self.n_gpu > 0:
        torch.cuda.manual_seed_all(self.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(self, train_dataset, model, tokenizer):
    """ Train the model """
    if self.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    self.train_batch_size = self.per_gpu_train_batch_size * max(1, self.n_gpu)
    train_sampler = RandomSampler(train_dataset) if self.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.train_batch_size)

    if self.max_steps > 0:
        t_total = self.max_steps
        self.num_train_epochs = self.max_steps // (len(train_dataloader) // self.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // self.gradient_accumulation_steps * self.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(self.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(self.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(self.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(self.model_name_or_path, "scheduler.pt")))

    if self.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=self.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if self.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if self.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", self.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", self.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        self.train_batch_size
        * self.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if self.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(self.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = self.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // self.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // self.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(self.num_train_epochs), desc="Epoch", disable=self.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(self)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(self.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if self.model_type in ["xlm", "roberta", "distilbert"]:
                del inputs["token_type_ids"]

            if self.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if self.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            if self.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if self.local_rank in [-1, 0] and self.logging_steps > 0 and global_step % self.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if self.local_rank == -1 and self.evaluate_during_training:
                        results = evaluate(self, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / self.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if self.local_rank in [-1, 0] and self.save_steps > 0 and global_step % self.save_steps == 0:
                    output_dir = os.path.join(self.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(self, os.path.join(output_dir, "training_self.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if self.max_steps > 0 and global_step > self.max_steps:
                epoch_iterator.close()
                break
        if self.max_steps > 0 and global_step > self.max_steps:
            train_iterator.close()
            break

    if self.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(self, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(self, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(self.output_dir) and self.local_rank in [-1, 0]:
        os.makedirs(self.output_dir)

    self.eval_batch_size = self.per_gpu_eval_batch_size * max(1, self.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.eval_batch_size)

    # multi-gpu evaluate
    if self.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", self.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(self.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if self.model_type in ["xlm", "roberta", "distilbert"]:
                del inputs["token_type_ids"]

            example_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if self.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(self.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(self.output_dir, "nbest_predictions_{}.json".format(prefix))

    if self.version_2_with_negative:
        output_null_log_odds_file = os.path.join(self.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if self.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            self.n_best_size,
            self.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            self.version_2_with_negative,
            tokenizer,
            self.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            self.n_best_size,
            self.max_answer_length,
            self.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            self.verbose_logging,
            self.version_2_with_negative,
            self.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results


def load_and_cache_examples(self, tokenizer, evaluate=False, output_examples=False):
    if self.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = self.data_dir if self.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, self.model_name_or_path.split("/"))).pop(),
            str(self.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not self.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not self.data_dir and ((evaluate and not self.predict_file) or (not evaluate and not self.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if self.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if self.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(self.data_dir, filename=self.predict_file)
            else:
                examples = processor.get_train_examples(self.data_dir, filename=self.train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=self.threads,
        )

        if self.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if self.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


class RunSquad:
    def __init__(self,
            model_type,
            model_name_or_path,
            output_dir,
            data_dir,
            train_file,
            predict_file,
            config_name = "",
            tokenizer_name = "",
            cache_dir = "",
            version_2_with_negative = True,
            null_score_diff_threshold = 0.0,
            max_seq_length = 384,
            doc_stride = 128,
            max_query_length = 100,
            do_train = True,
            do_eval = True,
            evaluate_during_training = True,
            do_lower_case = True,
            per_gpu_train_batch_size = 8,
            per_gpu_eval_batch_size = 8,
            learning_rate = 5e-5,
            gradient_accumulation_steps = 1,
            weight_decay = 0.0,
            adam_epsilon = 1e-8,
            max_grad_norm = 1.0,
            num_train_epochs = 3.0,
            max_steps = 1.0,
            warmup_steps = 0,
            n_best_size = 20,
            max_answer_length = 250,
            verbose_logging = True,
            logging_steps = 50,
            save_steps = 50,
            eval_all_checkpoints = True,
            no_cuda = True,
            overwrite_output_dir = True,
            overwrite_cache = True,
            seed = 42,
            local_rank = 1,
            fp16 = True,
            fp16_opt_level = "O1",
            server_ip = "",
            server_port = "",
            threads = 4
        ):
            self.model_type = model_type
            self.model_name_or_path = model_name_or_path
            self.output_dir = output_dir
            self.data_dir = data_dir
            self.train_file = train_file
            self.predict_file = predict_file
            self.config_name = config_name
            self.tokenizer_name = tokenizer_name
            self.cache_dir = cache_dir
            self.version_2_with_negative  = version_2_with_negative 
            self.null_score_diff_threshold = null_score_diff_threshold
            self.max_seq_length = max_seq_length
            self.doc_stride = doc_stride
            self.max_query_length  = max_query_length 
            self.do_train = do_train
            self.do_eval  = do_eval 
            self.evaluate_during_training = evaluate_during_training
            self.do_lower_case = do_lower_case
            self.per_gpu_train_batch_size = per_gpu_train_batch_size
            self.per_gpu_eval_batch_size  = per_gpu_eval_batch_size 
            self.learning_rate = learning_rate
            self.gradient_accumulation_steps = gradient_accumulation_steps
            self.weight_decay  = weight_decay 
            self.adam_epsilon  = adam_epsilon 
            self.max_grad_norm = max_grad_norm
            self.num_train_epochs = num_train_epochs
            self.max_steps  = max_steps 
            self.warmup_steps = warmup_steps
            self.n_best_size  = n_best_size 
            self.max_answer_length = max_answer_length
            self.verbose_logging  = verbose_logging 
            self.logging_steps  = logging_steps 
            self.save_steps  = save_steps 
            self.eval_all_checkpoints  = eval_all_checkpoints 
            self.no_cuda  = no_cuda 
            self.overwrite_output_dir  = overwrite_output_dir 
            self.overwrite_cache  = overwrite_cache 
            self.seed  = seed 
            self.local_rank  = local_rank 
            self.fp16  = fp16 
            self.fp16_opt_level = fp16_opt_level
            self.server_ip  = server_ip 
            self.server_port = server_port
            self.threads  = threads 

    def execute(self):
        if (
            os.path.exists(self.output_dir)
            and os.listdir(self.output_dir)
            and self.do_train
            and not self.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    self.output_dir
                )
            )

        # Setup distant debugging if needed
        if self.server_ip and self.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd

            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=(self.server_ip, self.server_port), redirect_output=True)
            ptvsd.wait_for_attach()

        # Setup CUDA, GPU & distributed training
        if self.local_rank == -1 or self.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1
        self.device = device

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if self.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            self.local_rank,
            device,
            self.n_gpu,
            bool(self.local_rank != -1),
            self.fp16,
        )

        # Set seed
        set_seed(self)

        # Load pretrained model and tokenizer
        if self.local_rank not in [-1, 0]:
            # Make sure only the first process in distributed training will download model & vocab
            torch.distributed.barrier()

        self.model_type = self.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_type]
        config = config_class.from_pretrained(
            self.config_name if self.config_name else self.model_name_or_path,
            cache_dir=self.cache_dir if self.cache_dir else None,
        )
        tokenizer = tokenizer_class.from_pretrained(
            self.tokenizer_name if self.tokenizer_name else self.model_name_or_path,
            do_lower_case=self.do_lower_case,
            cache_dir=self.cache_dir if self.cache_dir else None,
        )
        model = model_class.from_pretrained(
            self.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_name_or_path),
            config=config,
            cache_dir=self.cache_dir if self.cache_dir else None,
        )

        if self.local_rank == 0:
            # Make sure only the first process in distributed training will download model & vocab
            torch.distributed.barrier()

        model.to(self.device)

        logger.info("Training/evaluation parameters %s", self)

        # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if self.fp16 is set.
        # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
        # remove the need for this code, but it is still valid.
        if self.fp16:
            try:
                import apex

                apex.amp.register_half_function(torch, "einsum")
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        # Training
        if self.do_train:
            train_dataset = load_and_cache_examples(self, tokenizer, evaluate=False, output_examples=False)
            global_step, tr_loss = train(self, train_dataset, model, tokenizer)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Save the trained model and the tokenizer
        if self.do_train and (self.local_rank == -1 or torch.distributed.get_rank() == 0):
            # Create output directory if needed
            if not os.path.exists(self.output_dir) and self.local_rank in [-1, 0]:
                os.makedirs(self.output_dir)

            logger.info("Saving model checkpoint to %s", self.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(self.output_dir)
            tokenizer.save_pretrained(self.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self, os.path.join(self.output_dir, "training_self.bin"))

            # Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(self.output_dir)  # , force_download=True)
            tokenizer = tokenizer_class.from_pretrained(self.output_dir, do_lower_case=self.do_lower_case)
            model.to(self.device)

        # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
        results = {}
        if self.do_eval and self.local_rank in [-1, 0]:
            if self.do_train:
                logger.info("Loading checkpoints saved during training for evaluation")
                checkpoints = [self.output_dir]
                if self.eval_all_checkpoints:
                    checkpoints = list(
                        os.path.dirname(c)
                        for c in sorted(glob.glob(self.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                    )
                    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
            else:
                logger.info("Loading checkpoint %s for evaluation", self.model_name_or_path)
                checkpoints = [self.model_name_or_path]

            logger.info("Evaluate the following checkpoints: %s", checkpoints)

            for checkpoint in checkpoints:
                # Reload the model
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                model = model_class.from_pretrained(checkpoint)  # , force_download=True)
                model.to(self.device)

                # Evaluate
                result = evaluate(self, model, tokenizer, prefix=global_step)

                result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
                results.update(result)

        logger.info("Results: {}".format(results))

        return results

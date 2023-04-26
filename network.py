#!/usr/bin/env python
# -*- coding: utf-8 -*-
######################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# @file network.py
#
######################################################################
"""
File: network.py
"""

import os
import json
import logging
import argparse
import torch
from datetime import datetime
import numpy as np
import random
from transformers import BertTokenizer
# import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel


from source.inputters.corpus import KnowledgeCorpus
from source.models.dualcontrastive_seq2seq import DualContrastiveSeq2Seq
from source.models.seq2seq import Seq2Seq
from source.models.mem_seq2seq import MemSeq2Seq
from source.models.knowledge_seq2seq import KnowledgeSeq2Seq
from source.models.sequential_knowledge_transformer import SequentialKnowledgeTransformer

from source.utils.engine import Trainer
from source.utils.generator import TopKGenerator, BeamGenerator, GreedyGenerator, TopPGenerator
from source.utils.engine import evaluate, evaluate_generation
from source.utils.misc import str2bool


def model_config():
    """
    model_config
    """
    parser = argparse.ArgumentParser()

    # Data
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--data_dir", type=str, default="./data")
    data_arg.add_argument("--data_prefix", type=str, default="demo")
    data_arg.add_argument("--save_dir", type=str, default="./checkpoints")
    data_arg.add_argument("--output_dir", type=str, default="./output")
    data_arg.add_argument("--with_label", type=str2bool, default=False)
    data_arg.add_argument("--embed_file", type=str,
                          default="embeddings/glove.840B.300d.txt")
    data_arg.add_argument("--dataset", type=str, default="personachat")
    data_arg.add_argument("--model", type=str, default="knowledge_seq2seq")
    data_arg.add_argument("--stopwords", type=str, default="./stopwords.txt")
    data_arg.add_argument("--bert_dir", type=str, default="./pretrained_bert/uncased_L-12_H-768_A-12")

    # Network
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument("--embed_size", type=int, default=300)
    net_arg.add_argument("--hidden_size", type=int, default=800)
    net_arg.add_argument("--bidirectional", type=str2bool, default=True)
    net_arg.add_argument("--vocab_size", type=int, default=20000)
    net_arg.add_argument("--min_len", type=int, default=0)
    net_arg.add_argument("--max_len", type=int, default=100)
    net_arg.add_argument("--num_layers", type=int, default=1)
    net_arg.add_argument("--attn", type=str, default='dot',
                         choices=['none', 'mlp', 'dot', 'general'])
    net_arg.add_argument("--share_vocab", type=str2bool, default=True)
    net_arg.add_argument("--with_bridge", type=str2bool, default=True)
    net_arg.add_argument("--tie_embedding", type=str2bool, default=True)
    net_arg.add_argument("--num_heads", type=int, default=2)
    net_arg.add_argument("--filter_size", type=int, default=512)


    # Training / Testing
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument('--use_float16', type=str2bool, default=False)
    train_arg.add_argument('--enable_parallel', type=str2bool, default=True)
    train_arg.add_argument('--parallel', type=str2bool, default=False)
    train_arg.add_argument('--local_rank', type=int, default=0)
    train_arg.add_argument("--valid_metric_name", type=str, default="-loss")
    train_arg.add_argument("--optimizer", type=str, default="Adam")
    train_arg.add_argument("--lr", type=float, default=0.0005)
    train_arg.add_argument("--grad_clip", type=float, default=5.0)
    train_arg.add_argument("--dropout", type=float, default=0.3)
    train_arg.add_argument("--num_epochs", type=int, default=20)
    train_arg.add_argument("--pretrain_epoch", type=int, default=0)
    train_arg.add_argument("--lr_decay", type=float, default=None)
    train_arg.add_argument("--use_embed", type=str2bool, default=True)
    train_arg.add_argument("--use_dssm", type=str2bool, default=False)
    train_arg.add_argument("--use_pg", type=str2bool, default=False)
    train_arg.add_argument("--use_gs", type=str2bool, default=False)
    train_arg.add_argument("--use_kd", type=str2bool, default=False)
    train_arg.add_argument("--use_bow", type=str2bool, default=False)
    train_arg.add_argument("--use_posterior", type=str2bool, default=False)
    train_arg.add_argument("--posterior_detach", type=str2bool, default=True)
    train_arg.add_argument("--loss_form", type=str, default="kl_loss")
    train_arg.add_argument("--weight_control", type=str2bool, default=False)
    train_arg.add_argument("--decode_concat", type=str2bool, default=True)
    train_arg.add_argument("--multi_turn", type=str2bool, default=False)
    train_arg.add_argument("--alpha", type=float, default=0.05) # for prior contrastive
    train_arg.add_argument("--beta", type=float, default=0.05) # for post contrastive
    train_arg.add_argument("--gumbel_temperature", type=float, default=1.0)
    train_arg.add_argument("--use_copy_decoder", type=str2bool, default=False)


    # Geneation
    gen_arg = parser.add_argument_group("Generation")
    gen_arg.add_argument("--decoding_strategy", type=str, default='greedy_search',
                         choices=['greedy_search', 'beam_search', 'top_k', 'top_p'])
    gen_arg.add_argument("--beam_size", type=int, default=20)
    gen_arg.add_argument("--k", type=int, default=10)
    gen_arg.add_argument("--p", type=float, default=0.9)
    gen_arg.add_argument("--max_dec_len", type=int, default=30)
    gen_arg.add_argument("--ignore_unk", type=str2bool, default=True)
    gen_arg.add_argument("--length_average", type=str2bool, default=True)
    gen_arg.add_argument("--gold_score_file", type=str,
                         default="./gold.scores")

    # MISC
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument("--gpu", type=int, default=0)
    misc_arg.add_argument("--log_steps", type=int, default=100)
    misc_arg.add_argument("--valid_interval", type=str, default="steps", 
                          choices=["epoch", "steps"])
    misc_arg.add_argument("--valid_steps", type=int, default=100)
    misc_arg.add_argument("--batch_size", type=int, default=32)
    misc_arg.add_argument("--ckpt", type=str)
    #misc_arg.add_argument("--ckpt", type=str, default="models/best.model")
    misc_arg.add_argument("--check", action="store_true")
    misc_arg.add_argument("--test", action="store_true")
    misc_arg.add_argument("--interact", action="store_true")
    
    eval_arg = parser.add_argument_group("Evaluation")
    eval_arg.add_argument('-tns', '--train_source',
                        help='Path to the train source file, where each line ' +
                        'corresponds to one train input',
                        metavar='')
    eval_arg.add_argument('-tts', '--test_source',
                        help='Path to the test source file, where each line ' +
                        'corresponds to one test input',
                        metavar='')
    eval_arg.add_argument('-ttt', '--test_target',
                        help='Path to the test target file, where each line ' +
                        'corresponds to one test target',
                        metavar='')
    eval_arg.add_argument('-r', '--test_responses',
                        help='Path to the test model responses file',
                        metavar='')
    eval_arg.add_argument('-tv', '--text_vocab',
                        help='A file where each line is a word in the vocab',
                        metavar='')
    eval_arg.add_argument('-vv', '--vector_vocab',
                        help='A file where each line is a word in the vocab ' +
                        'followed by a vector',
                        metavar='')
    eval_arg.add_argument('-s', '--bleu_smoothing', default=4,
                        help='Bleu smoothing method (choices: %(choices)s)',
                        metavar='',
                        choices=[0, 1, 2, 3, 4, 5, 6, 7])
    eval_arg.add_argument('-t', '--t', default=1.97,
                        help='t value for confidence level calculation ' +
                        '(default: %(default)s)',
                        metavar='', type=int)

    config = parser.parse_args()

    return config


def main():
    """
    main
    """
    config = model_config()
    if config.check:
        config.save_dir = "./tmp"
    tokenizer = None
    if 'skt' in config.model:
        config.batch_size = 8
        if config.dataset.startswith('wizard'):
            config.batch_size = 4
            config.lr = 2.5*1e-4
        config.data_prefix = "bert"
        config.vocab_size = 30522
        config.grad_clip = 0.4
        # config.use_float16 = True
        tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
    # Data definition
    if torch.cuda.device_count()>1 and config.enable_parallel and (not config.test):
        config.use_gpu = True
        config.parallel = True
        torch.cuda.set_device(config.local_rank)
        torch.distributed.init_process_group(
            'nccl',
            init_method='env://'
        )
        device = f'cuda:{config.local_rank}'
    elif torch.cuda.device_count() > 0:
        config.use_gpu = True
        config.parallel = False
        device = f'cuda:{config.gpu}'
    else:
        config.use_gpu = False
        config.parallel = False
        device = 'cpu'
    torch.cuda.set_device(device)
    
    data_dir = os.path.join(config.data_dir, config.dataset)
    save_dir = os.path.join(config.save_dir, config.dataset, config.model)
    output_dir = os.path.join(config.output_dir, config.dataset, config.model)
    
    corpus = KnowledgeCorpus(data_dir=data_dir, data_prefix=config.data_prefix,
                             min_freq=0, vocab_size=config.vocab_size,
                             min_len=config.min_len, max_len=config.max_len,
                             embed_file=config.embed_file, with_label=config.with_label,
                             share_vocab=config.share_vocab, tokenizer=tokenizer)

    corpus.load()
    train_iter = corpus.create_dataloader(
        config.batch_size, "train", shuffle=True, device=device, parallel=config.parallel)
    valid_iter = corpus.create_dataloader(
        config.batch_size, "valid", shuffle=False, device=device, parallel=config.parallel)
    test_iter = corpus.create_dataloader(
        config.batch_size, "test", shuffle=False, device=device, parallel=config.parallel)

    # Model definition
    file_prefix = None
    ckpt_file = None
    if config.model == 'seq2seq':
        config.use_bow = False
        config.pretrain_epoch = 0
        config.use_posterior = False
        model = Seq2Seq(src_vocab_size=corpus.SRC.vocab_size,
                        tgt_vocab_size=corpus.TGT.vocab_size,
                        embed_size=config.embed_size, hidden_size=config.hidden_size,
                        padding_idx=corpus.padding_idx,
                        num_layers=config.num_layers, bidirectional=config.bidirectional,
                        attn_mode=config.attn, with_bridge=config.with_bridge,
                        tie_embedding=config.tie_embedding, dropout=config.dropout,
                        use_gpu=config.use_gpu)
    elif config.model.startswith('mem_seq2seq'):
        config.use_bow = False
        config.pretrain_epoch = 0
        config.use_posterior = False
        model = MemSeq2Seq(src_vocab_size=corpus.SRC.vocab_size,
                                 tgt_vocab_size=corpus.TGT.vocab_size,
                                 embed_size=config.embed_size, hidden_size=config.hidden_size,
                                 padding_idx=corpus.padding_idx,
                                 num_layers=config.num_layers, bidirectional=config.bidirectional,
                                 attn_mode=config.attn, with_bridge=config.with_bridge,
                                 tie_embedding=config.tie_embedding, dropout=config.dropout,
                                 use_gpu=config.use_gpu,
                                 use_bow=config.use_bow, use_dssm=config.use_dssm,
                                 use_pg=config.use_pg, use_gs=config.use_gs,
                                 pretrain_epoch=config.pretrain_epoch,
                                 weight_control=config.weight_control,
                                 concat=config.decode_concat,
                                 loss_form=config.loss_form)
    elif config.model.startswith('knowledge_seq2seq'):
        if config.model.endswith('pretrained_bow'):
            config.use_bow = True
            config.pretrain_epoch = 5
            config.num_epochs = config.pretrain_epoch
            config.use_posterior = True
        elif config.model.endswith('prior'):
            config.use_bow = False
            config.pretrain_epoch = 0
            config.use_posterior = False
        elif config.model.endswith('posterior'):
            config.use_bow = True
            config.pretrain_epoch = 5
            config.use_posterior = True
            if not config.test:
                ckpt = f"knowledge_seq2seq/pretrained_bow/state_epoch_{config.pretrain_epoch}"
                file_prefix = os.path.join(config.save_dir, config.dataset, ckpt)
                ckpt_file = file_prefix+'.model'
        model = KnowledgeSeq2Seq(src_vocab_size=corpus.SRC.vocab_size,
                                 tgt_vocab_size=corpus.TGT.vocab_size,
                                 embed_size=config.embed_size, hidden_size=config.hidden_size,
                                 padding_idx=corpus.padding_idx,
                                 num_layers=config.num_layers, bidirectional=config.bidirectional,
                                 attn_mode=config.attn, with_bridge=config.with_bridge,
                                 tie_embedding=config.tie_embedding, dropout=config.dropout,
                                 use_gpu=config.use_gpu,
                                 use_bow=config.use_bow, use_dssm=config.use_dssm,
                                 use_pg=config.use_pg, use_gs=config.use_gs,
                                 pretrain_epoch=config.pretrain_epoch,
                                 use_posterior=config.use_posterior,
                                 posterior_detach=config.posterior_detach,
                                 weight_control=config.weight_control,
                                 concat=config.decode_concat,
                                 loss_form=config.loss_form)
    elif config.model.startswith('skt_seq2seq'):
        config.multi_turn = True
        config.use_posterior = True
        config.hidden_size = 768
        model = SequentialKnowledgeTransformer(config)
    elif config.model.startswith('contrastive_seq2seq'):
        config.use_bow = True
        config.pretrain_epoch = 5
        config.use_posterior = True
        if not config.test:
            ckpt = f"knowledge_seq2seq/pretrained_bow/state_epoch_{config.pretrain_epoch}"
            file_prefix = os.path.join(config.save_dir, config.dataset, ckpt)
            ckpt_file = file_prefix+'.model'
        model = DualContrastiveSeq2Seq(src_vocab_size=corpus.SRC.vocab_size,
                                 tgt_vocab_size=corpus.TGT.vocab_size,
                                 embed_size=config.embed_size, hidden_size=config.hidden_size,
                                 padding_idx=corpus.padding_idx,
                                 num_layers=config.num_layers, bidirectional=config.bidirectional,
                                 attn_mode=config.attn, with_bridge=config.with_bridge,
                                 tie_embedding=config.tie_embedding, dropout=config.dropout,
                                 use_gpu=config.use_gpu,
                                 use_bow=config.use_bow, use_dssm=config.use_dssm,
                                 use_pg=config.use_pg, use_gs=config.use_gs,
                                 pretrain_epoch=config.pretrain_epoch,
                                 use_posterior=config.use_posterior,
                                 posterior_detach=config.posterior_detach,
                                 weight_control=config.weight_control,
                                 concat=config.decode_concat,
                                 loss_form=config.loss_form,
                                 alpha=config.alpha,
                                 beta=config.beta)
    
    # Generator definition
    if config.decoding_strategy == 'greedy_search':
        generator = GreedyGenerator(model=model, src_field=corpus.SRC, tgt_field=corpus.TGT, cue_field=corpus.CUE,
                                    max_length=config.max_dec_len, ignore_unk=config.ignore_unk,
                                    length_average=config.length_average, use_gpu=config.use_gpu, multi_turn=config.multi_turn)
    elif config.decoding_strategy == 'beam_search':
        generator = BeamGenerator(model=model, src_field=corpus.SRC, tgt_field=corpus.TGT, cue_field=corpus.CUE,
                                  beam_size=config.beam_size, max_length=config.max_dec_len, ignore_unk=config.ignore_unk,
                                  length_average=config.length_average, use_gpu=config.use_gpu, multi_turn=config.multi_turn)
    elif config.decoding_strategy == 'top_k':
        generator = TopKGenerator(model=model, src_field=corpus.SRC, tgt_field=corpus.TGT, cue_field=corpus.CUE,
                                  k=config.k, max_length=config.max_dec_len, ignore_unk=config.ignore_unk,
                                  length_average=config.length_average, use_gpu=config.use_gpu, multi_turn=config.multi_turn)
    elif config.decoding_strategy == 'top_p':
        generator = TopPGenerator(model=model, src_field=corpus.SRC, tgt_field=corpus.TGT, cue_field=corpus.CUE,
                                  p=config.p, max_length=config.max_dec_len, ignore_unk=config.ignore_unk,
                                  length_average=config.length_average, use_gpu=config.use_gpu, multi_turn=config.multi_turn)
    # Interactive generation testing
    if config.test and not config.ckpt:
        if not config.ckpt:
            setattr(config, "ckpt", "best")
    if config.ckpt:
        file_prefix = os.path.join(config.save_dir, config.dataset, config.model, config.ckpt)
        ckpt_file = file_prefix + '.model'
    if config.interact and ckpt_file:
        model.load(ckpt_file)
        return generator
    # Testing
    elif config.test:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        gen_file = os.path.join(output_dir, 'test.result')
        response_file = os.path.join(output_dir, 'test.responses')
        print(model)
        # trainer.load(file_prefix)
        model.load(ckpt_file)
        print("Testing ...")
        metrics = evaluate(model, test_iter, multi_turn=config.multi_turn, use_posterior=False)
        print(metrics.report_cum())
        print("Generating ...")
        stopwords = set(map(lambda x:x.rstrip('\n'), open(config.stopwords)))
        evaluate_generation(generator, 
                            test_iter,
                            save_file=gen_file, 
                            response_file=response_file,
                            verbos=True,
                            stopwords=stopwords)
        setattr(config, "train_source", f"data/{config.dataset}/train.src")
        setattr(config, "test_source", f"data/{config.dataset}/train.tgt")
        setattr(config, "test_target", f"data/{config.dataset}/test.tgt")
        setattr(config, "text_vocab", f"data/{config.dataset}/vocab.txt")
        setattr(config, "vector_vocab", f"data/{config.dataset}/vector.txt")
        setattr(config, "test_responses", f"output/{config.dataset}/{config.model}/test.responses")

    else:
        # Load word embeddings
        if 'skt' not in config.model and config.use_embed and config.embed_file is not None:
            model.enc_embedder.load_embeddings(
                corpus.SRC.embeddings, scale=0.03)
            model.dec_embedder.load_embeddings(
                corpus.TGT.embeddings, scale=0.03)
        # Optimizer definition
        optimizer = getattr(torch.optim, config.optimizer)(
            model.parameters(), lr=config.lr)
        # if config.use_float16:
        #     model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        if config.parallel:
            model = DistributedDataParallel(model, device_ids=[config.local_rank], find_unused_parameters=True)
        # Learning rate scheduler
        if config.lr_decay is not None and 0 < config.lr_decay < 1.0:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                      factor=config.lr_decay, patience=1, verbose=True, min_lr=1e-5)
        else:
            lr_scheduler = None
        # Save directory
        # date_str, time_str = datetime.now().strftime("%Y%m%d-%H%M%S").split("-")
        # result_str = "{}-{}".format(model_name, time_str)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Logger definition
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        log_file = os.path.join(save_dir, "train.log")
        with open(log_file, 'w') as f:
            f.truncate()
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)
        # Save config
        params_file = os.path.join(save_dir, "params.json")
        with open(params_file, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        print("Saved params to '{}'".format(params_file))
        logger.info(model)
        # Train
        logger.info("Training starts ...")
        trainer = Trainer(model=model, optimizer=optimizer, train_iter=train_iter,
                          valid_iter=valid_iter, test_iter=test_iter, logger=logger, generator=generator,
                          valid_metric_name=config.valid_metric_name, num_epochs=config.num_epochs,
                          save_dir=save_dir, log_steps=config.log_steps,
                          valid_interval=config.valid_interval, valid_steps=config.valid_steps, grad_clip=config.grad_clip,
                          lr_scheduler=lr_scheduler, save_summary=False, pretrain_epoch=config.pretrain_epoch, 
                          multi_turn=config.multi_turn, use_float16=config.use_float16, parallel=config.parallel, 
                          local_rank=config.local_rank, train_sampler=corpus.train_sampler)
        if file_prefix is not None:
            trainer.load(file_prefix=file_prefix)
        trainer.train()
        logger.info("Training done!")
        # Test
        logger.info("")
        trainer.load(os.path.join(save_dir, "best"))
        setattr(model, 'use_posterior', False)
        logger.info("Testing starts ...")
        metrics = evaluate(model, test_iter)
        logger.info(metrics.report_cum())
        logger.info("Generation starts ...")
        test_gen_file = os.path.join(save_dir, "test.result")
        stopwords = set(map(lambda x:x.rstrip('\n'), open(config.stopwords)))
        message = evaluate_generation(generator, test_iter,
                                      save_file=test_gen_file,
                                      stopwords=stopwords)
        logger.info(message)



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")

# -*- coding: utf-8 -*-

import codecs
import os

import torch
import torchtext

from onmt.io.DatasetBase import ONMTDatasetBase, PAD_WORD, BOS_WORD, EOS_WORD
import numpy as np

class VideoDataset(ONMTDatasetBase):
    """ Dataset for data_type=='video'

        Build `Example` objects, `Field` objects, and filter_pred function
        from video corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            tgt_seq_length (int): maximum target sequence length.
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """
    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 num_src_feats=0, num_tgt_feats=0,
                 tgt_seq_length=0, use_filter_pred=True):
        self.data_type = 'video'

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats

        if tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)
        out_examples = (self._construct_example_fromlist(
                            ex_values, out_fields)
                        for ex_values in example_values)
        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        out_examples = list(out_examples)

        def filter_pred(example):
            if tgt_examples_iter is not None:
                return 0 < len(example.tgt) <= tgt_seq_length
            else:
                return True

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(VideoDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    def sort_key(self, ex):
        """ shouldn't use this function"""
        return (256, 256)

    @staticmethod
    def make_video_examples_nfeats_tpl(path, vid_dir):
        """
        Args:
            path (str): location of a src file containing video paths
            src_dir (str): location of source videos

        Returns:
            (example_dict iterator, num_feats) tuple
        """
        examples_iter = VideoDataset.read_vid_file(path, vid_dir, 'src')
        num_feats = 0  # Source side(vid) has no features.

        return (examples_iter, num_feats)

    @staticmethod
    def read_vid_file(path, src_dir, side, truncate=None):
        """
        Args:
            path (str): location of a src file containing video paths
            src_dir (str): location of source videos
            side (str): 'src' or 'tgt'

        Yields:
            a dictionary containing video data, path and index for each line.
        """
        assert (src_dir is not None) and os.path.exists(src_dir),\
            'src_dir must be a valid directory if data_type is video'

        with codecs.open(path, "r", "utf-8") as corpus_file:
            index = 0
            for line in corpus_file:
                vid_name = line.split('.')[0]
                feat_path = os.path.join(src_dir, vid_name+'.npy')
                c3d_path = os.path.join(src_dir, vid_name+'_c3d.npy')
                feat = torch.from_numpy(np.load(feat_path))
                c3d = torch.from_numpy(np.load(c3d_path)).float()
                vid = torch.cat((feat, c3d.repeat(feat.shape[0], 1)), dim=1)

                example_dict = {side: vid,
                                side+'_path': line.strip(),
                                'indices': index}
                index += 1

                yield example_dict

    @staticmethod
    def get_fields(n_src_features, n_tgt_features):
        """
        Args:
            n_src_features: the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features: the number of target features to
                create `torchtext.data.Field` for.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        def make_vid(data, vocab, is_train):
            seq_len = data[0].size(0)
            dim_vid = data[0].size(1)
            vids = torch.zeros(len(data), seq_len, dim_vid)
            for i, vid in enumerate(data):
                vids[i] = vid
            return vids

        fields["src"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=make_vid, sequential=False)

        for j in range(n_src_features):
            fields["src_feat_"+str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        for j in range(n_tgt_features):
            fields["tgt_feat_"+str(j)] = \
                torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD,
                                     pad_token=PAD_WORD)

        def make_src(data, vocab, is_train):
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(src_size, len(data), src_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.FloatTensor,
            postprocessing=make_src, sequential=False)

        def make_tgt(data, vocab, is_train):
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[:sent.size(0), i] = sent
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            postprocessing=make_tgt, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, tensor_type=torch.LongTensor,
            sequential=False)

        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        For video corpus, source side is in form of video, thus
        no feature; while target side is in form of text, thus
        we can extract its text features.

        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.

        Returns:
            number of features on `side`.
        """
        if side == 'src':
            num_feats = 0
        else:
            with codecs.open(corpus_file, "r", "utf-8") as cf:
                f_line = cf.readline().strip().split()
                _, _, num_feats = VideoDataset.extract_text_features(f_line)

        return num_feats

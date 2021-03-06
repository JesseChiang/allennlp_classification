import logging
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
import jieba.posseg as poss
import jieba
from typing import *
from allennlp.data.tokenizers.word_splitter import WordSplitter
import os

from allennlp.data.tokenizers import Token
import csv

from typing import Dict

from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("text_classification_txt")
class TextClassificationTxtReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 delimiter: str = ',',
                 testing: bool = False,
                 max_sequence_length: int = None,
                 lazy: bool = False) -> None:
        """
        文本分类任务的datasetreader,从csv获取数据,head指定text,label.如:
        label   text
        sad    i like it.
        :param tokenizer: 分词器
        :param token_indexers:
        :param delimiter:
        :param testing:
        :param max_sequence_length:
        :param lazy:
        """
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._delimiter = delimiter
        self.testing = testing
        self._max_sequence_length = max_sequence_length

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, 'r') as inf:
            reader = csv.DictReader(inf)
            counter = 0
            for row in reader:
                counter += 1
                if self.testing and counter > 1000:
                    break
                title = str(row.get('title', ''))
                body = str(row.get('body', ''))
                text = title + '。' + body + str(row.get('text', ''))
                yield self.text_to_instance(text, str(row['label']))

    def _truncate(self, tokens):
        """
        truncate a set of tokens using the provided sequence length
        """
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[:self._max_sequence_length]
        return tokens

    @overrides
    def text_to_instance(self, text: str, label: Union[str, int] = None) -> Instance:
        """
        Parameters
        ----------
        text : ``str``, required.
            The text to classify
        label : ``str``, optional, (default = None).
            The label for this text.
        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The label label of the sentence or phrase.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        # tokenizer 默认使用了WordTokenizer, WordTokenizer有tokenize方法,返回词列表. splitter作为参数传给tokenizer
        text_tokens = self._tokenizer.tokenize(text)
        if self._max_sequence_length is not None:
            text_tokens = self._truncate(text_tokens)
        # TextField接收词列表和token_indexer,侯泽负责解决如何将词转换成编码.
        fields['tokens'] = TextField(text_tokens, self._token_indexers)
        if label:
            fields['label'] = LabelField(label)

        return Instance(fields)


@WordSplitter.register('jieba')
class JIEBASplitter(WordSplitter):
    """
    用jieba进行分词,可以定义用户词典和停用词词典.
    """

    def __init__(self, pos_tags: bool = False,
                 only_tokens: bool = True,
                 user_dict: str = None,
                 stop_words_path: str = None) -> None:
        #jieba.enable_parallel(4)
        self._pos_tags = pos_tags  # 是否标注词性。

        if user_dict and os.path.exists(user_dict):
            jieba.load_userdict(user_dict)
        self._only_tokens = only_tokens  # 最终是否只保留字符，去掉词性等属性

        self._stop_words = None  # 停用词
        if stop_words_path:
            self._stop_words = set()
            with open(stop_words_path, 'r') as f:
                for line in f:
                    word = line.strip()
                    self._stop_words.add(word)

    def _sanitize(self, tokens) -> List[Token]:
        """
        Converts spaCy tokens to allennlp tokens. Is a no-op if
        keep_spacy_tokens is True
        """
        sanitize_tokens = []
        if self._pos_tags:
            for text, pos in tokens:
                if self._stop_words and text in self._stop_words:
                    continue
                token = Token(text)
                if self._only_tokens:
                    pass
                else:
                    token = Token(token.text,
                                  token.idx,
                                  token.lemma_,
                                  pos,
                                  token.tag_,
                                  token.dep_,
                                  token.ent_type_)
                sanitize_tokens.append(token)
        else:
            for token in tokens:
                if self._stop_words and token in self._stop_words:
                    continue
                token = Token(token)
                sanitize_tokens.append(token)
        return sanitize_tokens

    @overrides
    def batch_split_words(self, sentences: List[str]) -> List[List[Token]]:
        split_words = []
        if self._pos_tags:
            for sent in sentences:
                split_words.append(self._sanitize(tokens) for tokens in poss.cut(sent))
        else:
            for sent in sentences:
                split_words.append(self._sanitize(tokens) for tokens in jieba.cut(sent))
        return split_words

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        if self._pos_tags:
            return self._sanitize(poss.cut(sentence))
        else:
            return self._sanitize(jieba.cut(sentence))

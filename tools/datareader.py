import logging
import re
from random import random
from typing import Dict, List
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides
from utils.helpers import SEQ_DELIMETERS, START_TOKEN

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("seq2edit_datareader")
class Seq2EditDatasetReader(DatasetReader):
    BROKEN_SENTENCES_REGEXP = re.compile(r'\.[a-zA-RT-Z]')
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 delimeters: dict = SEQ_DELIMETERS,
                 skip_correct: bool = False,
                 skip_complex: int = 0,
                 lazy: bool = False,
                 max_len: int = None,
                 test_mode: bool = False,
                 tag_strategy: str = "keep_one",
                 tn_prob: float = 0,
                 tp_prob: float = 0,
                 broken_dot_strategy: str = "keep") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers
        self._delimeters = delimeters
        self._max_len = max_len
        self._skip_correct = skip_correct
        self._skip_complex = skip_complex
        self._tag_strategy = tag_strategy
        self._broken_dot_strategy = broken_dot_strategy
        self._test_mode = test_mode
        self._tn_prob = tn_prob
        self._tp_prob = tp_prob

    @overrides
    def _read(self, file_path):
        file_path = cached_path(file_path)
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line or (not self._test_mode and self._broken_dot_strategy == 'skip'
                                and self.BROKEN_SENTENCES_REGEXP.search(line) is not None):
                    continue
                tokens_and_tags = [pair.rsplit(self._delimeters['labels'], 1)
                                   for pair in line.split(self._delimeters['tokens'])]
                try:
                    tokens = [Token(token) for token, tag in tokens_and_tags]
                    tags = [tag for token, tag in tokens_and_tags]
                except ValueError:
                        tokens = [Token(token) for token, tag in tokens_and_tags]
                        tags = [tag for token, tag in tokens_and_tags]

                if tokens and tokens[0] != Token(START_TOKEN):
                    tokens = [Token(START_TOKEN)] + tokens

                words = [x.text for x in tokens]
                if self._max_len is not None:
                    tokens = tokens[:self._max_len]
                    tags = None if tags is None else tags[:self._max_len]
                instance = self.text_to_instance(tokens, tags, words)
                if instance:
                    yield instance

    def extract_tags(self, tags: List[str]):
        op_del = self._delimeters['operations']

        labels = [x.split(op_del) for x in tags] 
        labels_final = []

        complex_flag_dict = {}
        # get flags
        for i in range(5):
            idx = i + 1
            complex_flag_dict[idx] = sum([len(x) > idx for x in labels]) 

        if self._tag_strategy == "keep_one":  
            for x in labels:  
                if len(x) == 1:
                    labels_final.append(x[0])
                elif len(x) > 5:
                    if self._skip_complex:
                        labels_final.append("$KEEP")
                    else:
                        labels_final.append(x[1] if x[0] == "$KEEP" else x[0])
                else:
                    labels_final.append(x[1] if x[0] == "$KEEP" else x[0])

        elif self._tag_strategy == "merge_all":
            pass
        else:
            raise Exception("Incorrect tag strategy")

        detect_tags = ["CORRECT" if label == "$KEEP" else "INCORRECT" for label in labels_final]  
        return labels_final, detect_tags, complex_flag_dict

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None,words: List[str] = None) -> Instance:  
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": words})
        if tags is not None:
            labels, detect_tags, complex_flag_dict = self.extract_tags(tags)
            if self._skip_complex: 
                if detect_tags.count("INCORRECT") / len(detect_tags) > 0.8:
                    return None
            rnd = random()
            if self._skip_correct and all(x == "CORRECT" for x in detect_tags):
                if rnd > self._tn_prob:
                    return None
            else:
                if rnd > self._tp_prob:
                    return None

            fields["labels"] = SequenceLabelField(labels, sequence,
                                                  label_namespace="labels")
            fields["d_tags"] = SequenceLabelField(detect_tags, sequence,
                                                  label_namespace="d_tags")
        return Instance(fields)

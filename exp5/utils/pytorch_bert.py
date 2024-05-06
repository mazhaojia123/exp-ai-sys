import logging
import numpy
import numpy as np
from utils.utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,SquadDatasetLoader,
                         RawResultExtended, write_predictions_extended, InputFeatures)
logger = logging.getLogger(__name__)

def load_and_cache_examples(predict_file, version_2_with_negative, max_seq_length, doc_stride, max_query_length, batch_size, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_file = predict_file
    logger.info("Creating features from dataset file at %s", input_file)
    examples = read_squad_examples(input_file=input_file,
                                            is_training=not evaluate,
                                            version_2_with_negative=version_2_with_negative)
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=max_seq_length,
                                            doc_stride=doc_stride,
                                            max_query_length=max_query_length,
                                            is_training=not evaluate)

    # MLU requires number of features must be a multiple of batch_size,
    # so we pad with fake features which are ignored later on.
    def pad_fake_feature():
        features.append(
            InputFeatures(
                unique_id=0,
                example_index=0,
                doc_span_index=0,
                tokens=features[-1].tokens,
                token_to_orig_map=features[-1].token_to_orig_map,
                token_is_max_context=features[-1].token_is_max_context,
                input_ids=[0] * max_seq_length,
                input_mask=[0] * max_seq_length,
                segment_ids=[0] * max_seq_length,
                cls_index=0,
                p_mask=[0] * max_seq_length,
                paragraph_len=0,
                start_position=None,
                end_position=None,
                is_impossible=False))

    while len(features) % batch_size != 0:
        pad_fake_feature()
        logger.info("  Pad one feature to eval_features, num of eval_features is %d now.", len(features))

    # Convert to Tensors and build dataset
    all_input_ids =   [f.input_ids for f in features]
    all_input_mask =  [f.input_mask for f in features]
    all_segment_ids = [f.segment_ids for f in features]
    all_cls_index =   [f.cls_index for f in features]
    all_p_mask =      [f.p_mask for f in features]
    
    all_input_ids =   np.array(all_input_ids) 
    all_input_mask =  np.array(all_input_mask) 
    all_segment_ids = np.array(all_segment_ids) 
    all_cls_index =   np.array(all_cls_index) 
    all_p_mask =      np.array(all_p_mask) 
    if evaluate:
        all_example_index = np.arange(len(features))
        dataset = (all_input_ids, all_input_mask, all_segment_ids, all_example_index, all_cls_index, all_p_mask)

    if output_examples:
        return dataset, examples, features
    return dataset

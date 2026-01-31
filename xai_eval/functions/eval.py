from .pred import class_proba

def hard_rationale_selection(token_weights, method='top_n', n=3, threshold=None):
    # Sort by weight descending, but keep index information
    sorted_token_weights = sorted(token_weights, key=lambda x: x[2], reverse=True)
    # Extract sorted indices and scores (ignore token)
    indices = [x[0] for x in sorted_token_weights]

    if method == 'top_n':
        if n is None:
            raise ValueError("Top-N method requires parameter n")
        important_positions = indices[:n]
    elif method == 'threshold':
        if threshold is None:
            raise ValueError("Threshold method requires parameter 'threshold'")
        # keep all tokens with score >= threshold
        important_positions = [x[0] for x in sorted_token_weights if x[2] >= threshold]
    else:
        raise ValueError(f"Invalid method: {method}")
    return important_positions

def comprehensivness(tokenizer, pipe, predicted_class, xai_token_importance, proba_dict, prediction_cache, method='top_n', n=3, threshold=None):
    predicted_class_proba = proba_dict[predicted_class]

    xai_token_importance2 = xai_token_importance.copy()

    tokens = [x[1] for x in xai_token_importance]
    important_positions = hard_rationale_selection(xai_token_importance2, method=method, n=n, threshold=threshold)

    # Remove tokens at those positions
    tokens_after_removal = [
        tok for i, tok in enumerate(tokens)
        if i not in important_positions
    ]
    comment_without_xai = tokenizer.convert_tokens_to_string(tokens_after_removal)
    
    if comment_without_xai not in prediction_cache:
        new_probability = class_proba(pipe, comment_without_xai)[predicted_class]
        prediction_cache[comment_without_xai] = new_probability
    else:
        new_probability = prediction_cache[comment_without_xai]

    return predicted_class_proba - new_probability

def sufficiency(tokenizer, pipe, predicted_class, xai_token_importance, proba_dict, prediction_cache, method='top_n', n=3, threshold=None):
    predicted_class_proba = proba_dict[predicted_class]

    xai_token_importance2 = xai_token_importance.copy()
    
    tokens = [x[1] for x in xai_token_importance]
    important_positions = hard_rationale_selection(xai_token_importance2, method=method, n=n, threshold=threshold)

    # only keep the important tokens
    important_tokens_only = [
        tok for i, tok in enumerate(tokens)
        if i in important_positions
    ]
    comment_with_xai_only = tokenizer.convert_tokens_to_string(important_tokens_only)

    # predict the comment with only the important tokens
    if comment_with_xai_only not in prediction_cache.keys():
        new_probability = class_proba(pipe, comment_with_xai_only)[predicted_class]
        prediction_cache[comment_with_xai_only] = new_probability
    else:
        new_probability = prediction_cache[comment_with_xai_only]

    return predicted_class_proba - new_probability
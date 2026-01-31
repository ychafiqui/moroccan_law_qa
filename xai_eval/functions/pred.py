

def predict_class(pipe, comment_content):
    probabilities = pipe(comment_content)[0]
    # Find the entry with the highest score
    return max(probabilities, key=lambda p: p["score"])["label"]

def class_proba(pipe, comment_content):
    probabilities = pipe(comment_content)[0]
    # Create a dictionary mapping labels to scores
    return {proba['label']: proba['score'] for proba in probabilities}
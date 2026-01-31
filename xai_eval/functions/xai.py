from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from lime import lime_text
import shap
from captum.attr import IntegratedGradients as CaptumIG
import torch
# import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

class LimeExplainer:
    def __init__(self, tokenizer, model, device, random_state=0):
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.random_state = random_state
        self.explainer = lime_text.LimeTextExplainer(class_names=list(self.model.config.label2id.keys()), 
                            split_expression=self.tokenize, feature_selection='none', random_state=self.random_state)
    
    def predictor(self, texts, batch_size=8, max_length=256):
        # texts is a list of strings from LIME
        all_probas = []
        with torch.inference_mode():  # <--- critical (no graph, no grad)
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,      # <--- critical to cap length
                    max_length=max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits
                probas = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probas.append(probas)

        return np.vstack(all_probas)

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        for i, token in enumerate(tokens):
            if token.startswith("##"):
                tokens[i] = tokens[i][2:]
        return tokens

    def explain(self, text, label, num_samples=50, normalize=True):
        label_id = self.model.config.label2id[label]
        tokens = self.tokenizer.tokenize(text)
        reconstructed_text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokens))
        
        # these two lines are used to make sure random state works
        self.explainer.random_state = np.random.RandomState(self.random_state)
        self.explainer.base.random_state = self.explainer.random_state
        
        start = time.time()
        exp = self.explainer.explain_instance(reconstructed_text, self.predictor, num_samples=num_samples, labels=[label_id])
        end = time.time()
        lime_time = end - start
        lime_exp = exp.as_list(self.model.config.label2id[label])
        lime_exp_dict = dict(lime_exp)

        if normalize:
            min_score = min(lime_exp_dict.values())
            max_score = max(lime_exp_dict.values())
            if max_score > 1 or min_score < -1:
                for token in lime_exp_dict.keys():
                    lime_exp_dict[token] = 2 * (lime_exp_dict[token] - min_score) / (max_score - min_score) - 1
        
        lime_exp = [[i, token, lime_exp_dict.get(token, lime_exp_dict.get(token[2:], 0))] for i, token in enumerate(tokens)]
        return lime_exp, lime_time
    
class ShapExplainer:
    def __init__(self, tokenizer, model, device, max_evals=50, random_state=0):
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        polarities = list(self.model.config.id2label.values())
        pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=device, top_k=None)
        self.explainer = shap.Explainer(pipe, self.tokenizer, algorithm="partition", output_names=polarities, 
                            max_evals=max_evals, seed=random_state)

    def explain(self, text, label, normalize=True):
        tokens = self.tokenizer.tokenize(text)
        start = time.time()
        shap_values = self.explainer([text])
        end = time.time()
        shap_time = end - start
        values = shap_values.values[0].tolist()[1:-1]
        polarity_idx = self.model.config.label2id[label]

        if normalize:
            min_score = min(values, key=lambda x: x[polarity_idx])[polarity_idx]
            max_score = max(values, key=lambda x: x[polarity_idx])[polarity_idx]
            if max_score > 1 or min_score < -1:
                values = [[token, 2 * (value[polarity_idx] - min_score) / (max_score - min_score) - 1] for token, value in zip(tokens, values)]
        
        shap_exp = []
        for i, (token, value) in enumerate(zip(tokens, values)):
            shap_exp.append([i, token, value[polarity_idx]])
        return shap_exp, shap_time

class IgExplainer:
    def __init__(self, tokenizer, model, device, max_length=256, n_steps=20, internal_batch_size=1):
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size
        
        def forward_func(input_embeds, attention_mask):
            # return probabilities (or logits) for Captum
            out = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
            return F.softmax(out.logits, dim=-1)

        self.forward_func = forward_func
        self.explainer = CaptumIG(self.forward_func)

    def explain(self, text, label, normalize=True):
        self.model.eval()

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # Build embeddings WITH grad tracking (Captum needs it)
        input_embeds = self.model.get_input_embeddings()(input_ids)
        baseline = torch.zeros_like(input_embeds)

        target = self.model.config.label2id[label]

        start = time.time()
        attributions = self.explainer.attribute(
            inputs=input_embeds,
            baselines=baseline,
            target=target,
            additional_forward_args=(attention_mask,),
            n_steps=self.n_steps,
            internal_batch_size=self.internal_batch_size,
        )
        end = time.time()

        # Token-level score: sum over embedding dims
        scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))[1:-1]

        if normalize and scores.size > 0:
            mn, mx = scores.min(), scores.max()
            if mx > 1 or mn < -1:
                scores = 2 * (scores - mn) / (mx - mn + 1e-12) - 1

        ig_exp = [[i, tok, float(score)] for i, (tok, score) in enumerate(zip(tokens, scores))]
        ig_time = end - start

        # Aggressive cleanup to avoid growth / fragmentation
        del enc, input_ids, attention_mask, input_embeds, baseline, attributions
        torch.cuda.empty_cache()

        return ig_exp, ig_time
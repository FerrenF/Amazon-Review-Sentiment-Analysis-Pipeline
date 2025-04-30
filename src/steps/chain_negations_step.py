import spacy
import logging
from core.step import Step

class ChainNegationsStep(Step):
    name = "chain_negations"

    def __init__(self, model="en_core_web_sm", max_chain_length=2, targets=["text"], disable=["parser", "ner"]):
        self.nlp = spacy.load(model, disable=disable)
        self.max_chain_length = max_chain_length
        self.negation_terms = {"not", "no", "never", "n't"}
        self.targets = targets

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]

        def chain_negations(text):
            if not isinstance(text, str):
                return text
            doc = self.nlp(text)
            tokens = []
            i = 0
            while i < len(doc):
                token = doc[i]
                if token.text.lower() in self.negation_terms:
                    chained_tokens = []
                    i += 1  # Skip the negation word
                    chain_count = 0
                    # Chain next up to N non-punctuation, non-space, non-stop tokens
                    while i < len(doc) and chain_count < self.max_chain_length:
                        next_token = doc[i]
                        if next_token.is_alpha and not next_token.is_stop:
                            chained_tokens.append(next_token.text.lower())
                            chain_count += 1
                        else:
                            break
                        i += 1
                    if chained_tokens:
                        tokens.append("not_" + "_".join(chained_tokens))
                else:
                    tokens.append(token.text.lower())
                    i += 1
            return " ".join(tokens)

        logging.info(f"Chaining negated words in {','.join(self.targets)} (up to {self.max_chain_length} tokens in length)...")
        for col in self.targets:
            if col in df.columns:
                df[col] = df[col].apply(chain_negations)

        logging.info("Negation chaining complete.")
        return data

import spacy
import logging
from core.step import Step

class ChainWordQualifiersStep(Step):
    name = "chain_qualifiers"

    def __init__(self, model="en_core_web_sm", max_chain_length=2, targets=["text"], disable=["parser", "ner"]):
        self.nlp = spacy.load(model, disable=disable)
        self.max_chain_length = max_chain_length

        # Expanded list of negators and intensifiers
        self.qualifier_terms = {
            # Negators
            "not", "no", "never", "n't", "nothing", "hardly", "barely", "scarcely", "little",
            # Intensifiers
            "very", "super", "extremely", "really", "totally", "utterly", "incredibly",
            "so", "highly", "deeply", "absolutely", "completely", "awfully",
            "especially", "ridiculously"
        }

        self.targets = targets

    def run(self, data: dict) -> dict:
        if "dataset" not in data:
            raise ValueError("No dataset found in data.")

        df = data["dataset"]

        def chain_qualifiers(text):
            if not isinstance(text, str):
                return text
            doc = self.nlp(text)
            tokens = []
            i = 0
            while i < len(doc):
                token = doc[i]
                if token.text.lower() in self.qualifier_terms:
                    chained_tokens = []
                    i += 1  # Move past qualifier
                    chain_count = 0
                    while i < len(doc) and chain_count < self.max_chain_length:
                        next_token = doc[i]
                        if next_token.is_alpha and not next_token.is_stop:
                            chained_tokens.append(next_token.text.lower())
                            chain_count += 1
                        else:
                            break
                        i += 1
                    if chained_tokens:
                        chained_token = token.text.lower() + "_" + "_".join(chained_tokens)
                        tokens.append(chained_token)
                    else:
                        tokens.append(token.text.lower())
                else:
                    tokens.append(token.text.lower())
                    i += 1
            return " ".join(tokens)

        logging.info(f"Chaining qualifiers in {','.join(self.targets)} (up to {self.max_chain_length} tokens)...")
        for col in self.targets:
            if col in df.columns:
                df[col] = df[col].apply(chain_qualifiers)

        logging.info("Qualifier chaining complete.")
        return data

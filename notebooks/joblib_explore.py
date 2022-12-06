import pandas as pd
import spacy
import pytextrank
from spacy.language import Language
from spacy_readability import Readability
from importlib.util import find_spec
import gensim.downloader as api
from itertools import starmap, combinations
from multiprocessing import cpu_count
from joblib import Parallel, delayed


@Language.component("readability")
def readability(doc):
    read = Readability()
    doc = read(doc)
    return doc


def apply_ranking(doc, trt):
    results = []
    for phrase in doc._.phrases:
        if phrase.rank >= trt:
            results.append((phrase.text, phrase.rank))
    return results


def apply_readability(doc):
    return doc._.flesch_kincaid_reading_ease


# load pipelines and models
print("loading pipelines and models...")
pipe_name = "en_core_web_md"
model_name = "word2vec-google-news-300"

if find_spec(pipe_name) is None:
    spacy.cli.download(pipe_name)
nlp = spacy.load(pipe_name)
nlp.add_pipe("textrank", last=True)
nlp.add_pipe("readability", last=True)
stopwords = list(nlp.Defaults.stop_words)
model = api.load(model_name)

# load input data
print("loading data...")
dpath = "../data/sample_input.json"
df = pd.read_json(dpath, lines=True)
df_reviews = df.loc[df.astype("str").drop_duplicates().index]

# compute token ranks and argument readabilities
print("Computing ranks and readabilities...")
scores = []
docs = nlp.pipe(texts=df_reviews["reviewText"].astype("str"))
for doc in docs:
    scores.append([apply_ranking(doc, 0), apply_readability(doc)])

df_reviews["ranks"] = [p[0] for p in scores]
df_reviews["n_tokens"] = [len(p[0]) for p in scores]
df_reviews["readability"] = [p[1] for p in scores]
df_reviews.sort_values(by="n_tokens", ascending=False)

# get tokens
print("Arranging tokens...")
tokens = []
for phrases_rank in list(df_reviews["ranks"]):
    for phrase in phrases_rank:
        phrase = phrase[0].lower().split()
        phrase = filter(lambda t: t not in stopwords, phrase)
        phrase = " ".join(phrase)
        if phrase:
            tokens.append(phrase)

combis = list(combinations(tokens, 2))


def calc_wmdistance(model, docs):
    return model.wmdistance(docs[0], docs[1])


print("Computing distance matrix...")
dists = Parallel(n_jobs=3, backend="loky", verbose=100)(
    delayed(calc_wmdistance)(model, combi) for combi in combis
)

print(dists[:10])

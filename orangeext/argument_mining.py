"""
Argument mining module.

Input: json file that contains a collection of arguments on the same product, together with an overall score of each
Output: a networkx instance that describe the attacking network of the arguments.

Author: @jiqicn
"""

import pandas as pd
import spacy
import pytextrank
from spacy.language import Language
from spacy_readability import Readability
from importlib.util import find_spec
import gensim.downloader as api
import numpy as np
from itertools import starmap, combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


@Language.component("readability")
def readability(doc):
    read = Readability()
    doc = read(doc)
    return doc


class ArgumentMiner(object):
    def __init__(self, fpath: str):
        df = pd.read_json(fpath, lines=True)
        self.df_arguments = df.loc[df.astype(str).drop_duplicates().index]

    def load_nlp_pipeline(self, pipe_name: str = "en_core_web_md"):
        """
        Load the NLP pipeline that is built in spacy package.
        Will download the pipeline files if not exist.
        """
        if find_spec(pipe_name) is None:
            spacy.cli.download(pipe_name)

        self.nlp_pipe = spacy.load(pipe_name)
        self.nlp_pipe.add_pipe("textrank", last=True)
        self.nlp_pipe.add_pipe("readability", last=True)

    def load_word_vector_model(self, model_name: str = "word2vec-google-news-300"):
        """
        Load the word vector model that is built in gensim package.
        Will download the model files if not exist.
        """
        self.wv_model = api.load(model_name)

    @staticmethod
    def __get_token_and_rank(doc: spacy.tokens.Doc, stopwords: list, trt: float = 0):
        """
        Get text rank of each token.
        """
        results = []
        for token in doc._.phrases:
            text = token.text
            text = text.lower().split(" ")
            text = filter(lambda x: x not in stopwords, text)
            text = " ".join(text)
            if token.rank and token.rank >= trt:
                results.append((text, token.rank))
        return results

    @staticmethod
    def __get_doc_readability(doc: spacy.tokens.Doc):
        """
        Get readability score of a document.
        """
        return doc._.flesch_kincaid_reading_ease

    def compute_ranks_and_readability(self):
        """
        For each argument in the input dataset, compute the token text ranks and readability.
        These data will be addd to the arguments dataframe as two columns.
        """
        stopwords = list(self.nlp_pipe.Defaults.stop_words)
        ranks = []
        readabilities = []
        docs = self.nlp_pipe.pipe(texts=self.df_arguments["reviewText"].astype("str"))
        for doc in docs:
            ranks.append(self.__get_token_and_rank(doc, stopwords, 0))
            readabilities.append(self.__get_doc_readability(doc))
        self.df_arguments["ranks"] = ranks
        self.df_arguments["readability"] = readabilities

    def get_all_tokens(self):
        """
        Get the full list of tokens in the arguments
        """
        tokens = []
        for doc in list(self.df_arguments["ranks"]):
            for token in doc:
                token = token[0]
                if token:
                    tokens.append(token)
        tokens = np.array(tokens)

        return tokens

    def __get_token_distance_matrix(self, tokens: np.array):
        """
        Cluster tokens on the compute distance matrix by KMeans, return the cluster label list
        """
        token_pairs = list(combinations([t.split(" ") for t in tokens], 2))
        token_dists = list(starmap(self.wv_model.wmdistance, token_pairs))
        token_dists = np.nan_to_num(token_dists, nan=0, posinf=100)

        # mirror distances along the diagonal of distance matrix
        dist_matrix = np.zeros((len(tokens), len(tokens)))
        dist_matrix[np.triu_indices(len(tokens), 1)] = token_dists
        dist_matrix = dist_matrix + dist_matrix.T
        dist_matrix = pd.DataFrame(dist_matrix, index=tokens, columns=tokens)

        return dist_matrix

    @staticmethod
    def __cluster(dist_matrix: pd.DataFrame, k: int):
        """
        cluster the items involved in dist_matrix in k parts.
        """
        cluster = KMeans(n_clusters=k, random_state=10)
        labels = cluster.fit_predict(dist_matrix)
        try:
            silhouette = silhouette_score(dist_matrix, labels)
        except:
            silhouette = 0
        return silhouette, labels

    def get_cluster_labels(self, tokens: np.array):
        """
        get clusters of tokens
        """
        dist_matrix = self.__get_token_distance_matrix(tokens)
        cluster_labels = None
        silhouette_target = -float("inf")
        for i in range(min(dist_matrix.index.size, 10)):
            silhouette, labels = self.__cluster(dist_matrix, i + 1)
            if silhouette > silhouette_target:
                silhouette_target = silhouette
                cluster_labels = labels

        return cluster_labels

    @staticmethod
    def __get_cluster_set(
        tokens: np.array, token_dictionary: np.array, cluster_labels: np.array
    ):
        """
        Find the cluster labels of all tokens in a token list and return it as a set
        """
        indices = np.isin(token_dictionary, tokens)
        clusters = cluster_labels[indices]
        return set(clusters.flatten())

    # TODO: Tasks pending completion -@jiqi at 11/29/2022, 4:43:27 PM
    # refactor the create_netowrk function to split it into atomic functions
    def create_network(self, tokens: np.array, cluster_labels: np.array, theta: int):
        """
        Compute the attacking relations between arguments and create the attacking network.
        """
        source = []
        destination = []
        weight = []

        # form attacking relations
        for curr_group in range(1, 5):
            # cond: arguments with different overall score
            df_src = self.df_arguments[self.df_arguments["overall"] == curr_group]
            df_dst = self.df_arguments[self.df_arguments["overall"] > curr_group]

            for i_src, row_src in df_src.iterrows():
                if not row_src["ranks"]:
                    continue
                query_src = np.array([t[0] for t in row_src["ranks"]])
                clusters_src = self.__get_cluster_set(query_src, tokens, cluster_labels)
                w_src = max([t[1] for t in row_src["ranks"]]) * row_src["readability"]

                for i_dst, row_dst in df_dst.iterrows():
                    query_dst = np.array([t[0] for t in row_dst["ranks"]])
                    clusters_dst = self.__get_cluster_set(
                        query_dst, tokens, cluster_labels
                    )

                    # cond: arguments sharing tokens from the same cluster
                    if clusters_src and clusters_dst:
                        w_dst = (
                            max([t[1] for t in row_dst["ranks"]])
                            * row_dst["readability"]
                        )

                        # attack from high-weight argument to low-weight one
                        if w_src >= w_dst:
                            source.append(i_src)
                            destination.append(i_dst)
                            weight.append(w_src)
                        else:
                            source.append(i_dst)
                            destination.append(i_src)
                            weight.append(w_dst)

        # create network instance
        df_network = pd.DataFrame(
            {"source": source, "destination": destination, "weight": weight}
        )
        df_network = df_network[self.df_network["weight"] >= theta]  # pruning strategy
        G = nx.from_pandas_edgelist(
            df_network,
            source="source",
            target="destination",
            edge_attr=["weight"],
            create_using=nx.DiGraph(),
        )

        # label nodes as supported (green), defeated (red), and undecided (grey)
        nodes = list(G.nodes)
        colors = np.full(shape=len(nodes), fill_value="grey", dtype=object)

        supported = []
        targets = set(df_network["destination"].unique())
        for i, node in enumerate(nodes):
            attackers = set(df_network[df_network["destination"] == node]["source"])

            # cond: no attacker or all attackers are under attacked
            if (not attackers) or (attackers & targets == attackers):
                colors[i] = "green"
                supported.append(node)
        supported = set(supported)

        defeated = []
        for i in np.where(colors == "grey")[0]:
            node = nodes[i]
            if node in targets:
                attackers = set(df_network[df_network["destination"] == node]["source"])

                # cond: if under attacked by 'supported' nodes
                if attackers & supported:
                    colors[i] = "red"
                    defeated.append(node)

        attr_colors = {}
        for i, node in enumerate(nodes):
            attr_colors[node] = colors[i]
        nx.set_node_attributes(G, attr_colors, name="color")

        return G


if __name__ == "__main__":
    fpath = "../data/sample_input.json"
    am = ArgumentMiner(fpath)

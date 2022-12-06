from orangeext.argument_mining import ArgumentMiner
import pytest
import os


@pytest.fixture
def miner():
    basePath = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.join(basePath, "test_data.json")
    miner = ArgumentMiner(fpath)
    miner.load_nlp_pipeline()
    miner.load_word_vector_model()
    return miner


def test_compute_ranks_readability(miner):
    theta = 0
    miner.compute_ranks_and_readability(token_theta=theta)
    assert "ranks" in miner.df_arguments.columns
    assert "readability" in miner.df_arguments.columns

    tokens = miner.df_arguments.loc[0, "ranks"]
    assert len(tokens) > 0
    assert all(t[0] != "" for t in tokens)
    assert all(t[1] >= theta for t in tokens)


def test_compute_clusters_and_weights(miner):
    theta = 0
    miner.compute_ranks_and_readability(token_theta=theta)
    miner.compute_clusters_and_weights()
    assert "clusters" in miner.df_arguments.columns
    assert "weight" in miner.df_arguments.columns

    clusters_list = list(miner.df_arguments["clusters"])
    assert set.intersection(*clusters_list)


def test_compute_network_and_labels(miner):
    theta = 0
    miner.compute_ranks_and_readability(token_theta=theta)
    miner.compute_clusters_and_weights()
    miner.compute_network(weight_theta=theta)
    miner.compute_network_node_colors()

    assert len(miner.network.nodes) > 0
    assert miner.network.edges[1, 0]
    assert "weight" in miner.network.edges[1, 0]
    with pytest.raises(KeyError):
        miner.network.edges[0, 1]

    n0 = miner.network.nodes[0]
    n1 = miner.network.nodes[1]
    n2 = miner.network.nodes[2]
    assert n0["color"] == "red"
    assert n1["color"] == "red"
    assert n2["color"] == "green"

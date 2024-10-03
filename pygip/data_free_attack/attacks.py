from .generator import GraphGenerator
from .utils import GraphNeuralNetworkMetric

class DataFreeModelExtractionAttack:
    def __init__(self, victim_model, graph, features, labels, attack_type=0):
        self.victim_model = victim_model
        self.graph = graph
        self.features = features
        self.labels = labels
        self.attack_type = attack_type
        self.generator = GraphGenerator(features.shape[1], graph.number_of_nodes())

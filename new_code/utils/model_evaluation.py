import torch as th
from scipy.special import softmax


def compute_accuracy(predictions, labels):
    """
    Compute the accuracy of predictions given the labels.

    Args:
        predictions (torch.Tensor): The predicted values.
        labels (torch.Tensor): The true labels.

    Returns:
        float: The accuracy of the predictions.
    """
    labels = labels.long()
    correct_predictions = (th.argmax(predictions, dim=1) == labels).float()
    accuracy = correct_predictions.sum() / len(predictions)
    return accuracy


def evaluate_model(model, graph, inputs, labels, validation_node_ids, device):
    """
    Evaluate the model on the validation set.

    Args:
        model: The model to be evaluated.
        graph (DGLGraph): The entire graph.
        inputs (torch.Tensor): The features of all the nodes.
        labels (torch.Tensor): The labels of all the nodes.
        validation_node_ids (list): List of node IDs for validation.
        device: The device to run the evaluation on (CPU or GPU).

    Returns:
        tuple: The accuracy of the model and the softmax probabilities of the predictions.
    """
    model.eval()  # Set the model to evaluation mode
    with th.no_grad():  # Disable gradient computation
        predictions = model.inference(graph, inputs, device)

    model.train()  # Switch back to training mode
    accuracy = compute_accuracy(predictions[validation_node_ids], labels[validation_node_ids])
    softmax_probs = softmax(predictions[validation_node_ids].detach().cpu().numpy(), axis=1)

    return accuracy, softmax_probs

import torch
import math


class ListMLE(torch.nn.Module):
    def __init__(self, eps=1e-10, padded_value_indicator=-1):
        super(ListMLE, self).__init__()
        self.eps = eps
        self.padded_value_indicator = padded_value_indicator

    def forward(self, y_pred, y_true):
        return list_MLE(y_pred, y_true, self.eps, self.padded_value_indicator)


def list_MLE(y_pred, y_true, eps=1e-10, padded_value_indicator=-1):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(
        preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1
    ).flip(dims=[1])

    DCG = math.prod(1 / math.log2(i + 1) for i in range(1, len(indices)))
    observation_loss = (torch.log(cumsums + eps) - preds_sorted_by_true_minus_max)*DCG

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))

if __name__ == "__main__":
    y_pred = torch.rand(5,5)
    y_true = y_pred.t()
    random_indices = torch.randperm(y_true.shape[-1])
    print(random_indices)
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]
    print("y_pred:\n",y_pred)
    print("y_true:\n",y_true)
    print("y_pred_shuffled:\n",y_pred_shuffled)
    print("y_true_shuffled:\n",y_true_shuffled)
    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
    print("y_true_sorted:\n",y_true_sorted)
    print("indices:\n",indices)
    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    print("preds_sorted_by_true:\n",preds_sorted_by_true)
    print("max_pred_values:\n",max_pred_values)


import torch.optim as optim


def load_optimizer(optimizer, model, learning_rate, momentum):
    if optimizer == 'adam':
        model_optimizer = optim.Adam(model.parameters(),
                                     lr=learning_rate)
    elif optimizer == 'sgd':
        model_optimizer = optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    momentum=momentum,
                                    nesterov=True)
    return model_optimizer


if __name__ == '__main__':
    pass

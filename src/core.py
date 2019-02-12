import os
import pickle

import numpy as np
import torch

from utils import noisy


def train(trainset, model, optimizer, criterion, batch_size, device, epoch,
          example_stats, epochs, train_indx):
    """ Train model for one epoch.

    :param example_stats: Dictionary containing statistics accumulated over
    every presentation of example.
    """
    train_loss = 0
    correct = 0
    total = 0

    model.train()

    trainset_permutation_inds = np.random.permutation(
        np.arange(len(trainset.train_labels)))

    for batch_idx, batch_start_ind in \
            enumerate(range(0, len(trainset.train_labels), batch_size)):

        batch_inds = trainset_permutation_inds[batch_start_ind:
                                               batch_start_ind + batch_size]

        transformed_trainset = []
        for ind in batch_inds:
            transformed_trainset.append(trainset.__getitem__(ind)[0])
        inputs = torch.stack(transformed_trainset)
        targets = torch.LongTensor(
            np.array(trainset.train_labels)[batch_inds].tolist())

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)

        acc = predicted == targets
        for j, index in enumerate(batch_inds):
            index_in_original_dataset = train_indx[index]

            output_correct_class = outputs.data[
                j, targets[j].item()]
            sorted_output, _ = torch.sort(outputs.data[j, :])
            if acc[j]:
                output_highest_incorrect_class = sorted_output[-2]
            else:
                output_highest_incorrect_class = sorted_output[-1]
            margin = output_correct_class.item(
            ) - output_highest_incorrect_class.item()

            index_stats = example_stats.get(index_in_original_dataset,
                                            [[], [], []])
            index_stats[0].append(loss[j].item())
            index_stats[1].append(acc[j].sum().item())
            index_stats[2].append(margin)
            example_stats[index_in_original_dataset] = index_stats

        loss = loss.mean()
        train_loss += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        loss.backward()
        optimizer.step()
        index_stats = example_stats.get('train', [[], []])
        index_stats[1].append(100. * correct.item() / float(total))
        example_stats['train'] = index_stats


def test(testset, model, criterion, device, example_stats, epoch, output_dir,
         dataset_name, save_fname, best_acc):
    """ Evaluate model predictions on heldout test data.

    :param example_stats: Dictionary containing statistics accumulated over
    every presentation of example.
    """
    test_loss = 0
    correct = 0
    total = 0
    test_batch_size = 32

    model.eval()

    for batch_idx, batch_start_ind in enumerate(
            range(0, len(testset.test_labels), test_batch_size)):
        transformed_testset = []
        for ind in range(
                batch_start_ind,
                min(
                    len(testset.test_labels),
                    batch_start_ind + test_batch_size)):
            transformed_testset.append(testset.__getitem__(ind)[0])
        inputs = torch.stack(transformed_testset)
        targets = torch.LongTensor(
            np.array(testset.test_labels)[batch_start_ind:batch_start_ind +
                                          test_batch_size].tolist())

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss = loss.mean()
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100. * correct.item() / total
    index_stats = example_stats.get('test', [[], []])
    index_stats[1].append(100. * correct.item() / float(total))
    example_stats['test'] = index_stats
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %
          (epoch, loss.item(), acc))

    # Save best model to checkpoint dir
    if acc > best_acc:
        state = {
            'acc': acc,
            'epoch': epoch,
        }
        save_point = os.path.join(output_dir, 'checkpoint', dataset_name)
        os.makedirs(save_point, exist_ok=True)
        torch.save(state, os.path.join(save_point, save_fname + '.t7'))
        best_acc = acc


def remove_examples(trainset, removing_method, remove_n, remove_subsample,
                    sorting_file, input_dir):
    removing_method = -1 if removing_method == '1' else 0
    if sorting_file == 'none':
        train_indx = np.array(range(len(trainset.train_labels)))
    else:
        try:
            with open(os.path.join(input_dir, sorting_file) + '.pkl', 'rb') \
                    as fin:
                ordered_indx = pickle.load(fin)['indices']
        except IOError:
            with open(os.path.join(input_dir, sorting_file), 'rb') as fin:
                ordered_indx = pickle.load(fin)['indices']
        elements_to_remove = np.array(
            ordered_indx)[removing_method:removing_method + remove_n]
        train_indx = np.setdiff1d(range(len(trainset.train_labels)),
                                  elements_to_remove)

    # Remove samples at random
    if removing_method < 0:
        train_indx = np.random.permutation(np.arange(len(
            trainset.train_labels)))[:len(trainset.train_labels) - remove_n]
    elif remove_subsample:
        lowest_n = np.array(ordered_indx)[0:removing_method]
        train_indx = lowest_n[np.random.permutation(np.arange(
            removing_method))[:removing_method - remove_subsample]]
        train_indx = np.hstack((train_indx,
                                np.array(ordered_indx)[removing_method:]))
    trainset.train_data = trainset.train_data[train_indx, :, :]
    trainset.train_labels = \
        np.array(trainset.train_labels)[train_indx].tolist()
    return train_indx


def apply_noise(trainset, train_indx, noise_percent_pixels,
                noise_percent_labels, noise_std_pixels, output_dir,
                save_fname):
    # Noise img
    if noise_percent_pixels:
        for i in range(len(train_indx)):
            img = trainset.train_data[i, :, :, :]
            noisy_img = noisy(img, noise_percent_pixels, noise_std_pixels)
            trainset.train_data[i, :, :, :] = noisy_img

    # Noise labels
    if noise_percent_labels:
        fname = os.path.join(output_dir, save_fname)

        with open(fname + "_changed_labels.txt", "w") as f:
            n_labels_to_change = int(
                noise_percent_labels * len(trainset.train_labels) / 100)
            labels_idx_to_change = np.random.choice(
                np.arange(len(trainset.train_labels)),
                n_labels_to_change, replace=False)

            # Flip random labels
            for l, ind_l in enumerate(labels_idx_to_change):
                l_choices = np.arange(
                    len(np.unique(trainset.train_labels)))
                true_l = trainset.train_labels[ind_l]
                l_choices = np.delete(l_choices, true_l)
                noisy_label = np.random.choice(l_choices, 1)
                trainset.train_labels[ind_l] = noisy_label[0]
                f.write(
                    str(train_indx[ind_l]) + ' ' +
                    str(true_l) + ' ' + str(noisy_label[0]) + '\n')


if __name__ == '__main__':
    pass

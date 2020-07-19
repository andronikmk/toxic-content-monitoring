import copy
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from nni.nas.pytorch.trainer import Trainer
from nni.nas.pytorch.utils import AverageMeterGroup

from nni.nas.pytorch.darts.mutator import DartsMutator

logger = logging.getLogger(__name__)

class DartsTrainer(Trainer):
    """
    DARTS trainer.
    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained.
    loss : callable
        Receives logits and ground truth label, return a loss tensor.
    metrics : callable
        Receives logits and ground truth label, return a dict of metrics.
    optimizer : Optimizer
        The optimizer used for optimizing the model.
    num_epochs : int
        Number of epochs planned for training.
    dataset_train : Dataset
        Dataset for training. Will be split for training weights and architecture weights.
    dataset_valid : Dataset
        Dataset for testing.
    mutator : DartsMutator
        Use in case of customizing your own DartsMutator. By default will instantiate a DartsMutator.
    batch_size : int
        Batch size.
    workers : int
        Workers for data loading.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    log_frequency : int
        Step count per logging.
    callbacks : list of Callback
        list of callbacks to trigger at events.
    arc_learning_rate : float
        Learning rate of architecture parameters.
    unrolled : float
        ``True`` if using second order optimization, else first order optimization.
    """
    def __init__(self, model, loss, metrics,
                 optimizer, num_epochs, dataset_train, dataset_valid,
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None,
                 callbacks=None, arc_learning_rate=3.0E-4, unrolled=False):
        super().__init__(model, mutator if mutator is not None else DartsMutator(model),
                         loss, metrics, optimizer, num_epochs, dataset_train, dataset_valid,
                         batch_size, workers, device, log_frequency, callbacks)

        print("Mutator Params:", len(list(self.mutator.parameters())))

        self.ctrl_optim = torch.optim.Adam(self.mutator.parameters(), arc_learning_rate, betas=(0.5, 0.999),
                                           weight_decay=1.0E-3)
        self.unrolled = unrolled

        X_train = [x[0] for x in dataset_train]
        y_train = [x[1] for x in dataset_train]
    
        batched_train_X = []
        batched_train_y = []

        i=0
        while i < len(dataset_train) // (2 * batch_size):
            batched_train_X.append(X_train[i*batch_size:(i+1)*batch_size])
            batched_train_y.append(y_train[i*batch_size:(i+1)*batch_size])
            i+=1
            
        self.batched_train = list(zip(batched_train_X.copy(), batched_train_y.copy()))

        batched_validate_X = []
        batched_validate_y = []

        while i+1 < len(dataset_train) // batch_size:
            batched_validate_X.append(X_train[i*batch_size:(i+1)*batch_size])
            batched_validate_y.append(y_train[i*batch_size:(i+1)*batch_size])
            i+=1
        
        self.batched_validate = list(zip(batched_validate_X.copy(), batched_validate_y.copy()))

        X_test = [x[0] for x in dataset_valid]
        y_test = [x[1] for x in dataset_valid] 
        
        batched_test_X = []
        batched_test_y = []

        i=0
        while i+1 < len(dataset_valid) // batch_size:
            batched_test_X.append(X_test[i*batch_size:(i+1)*batch_size])
            batched_test_y.append(y_test[i*batch_size:(i+1)*batch_size])
            i+=1
        del i

        self.batched_test = list(zip(batched_test_X.copy(), batched_test_y.copy()))

    def train_one_epoch(self, epoch):
        self.mutator.train()
        meters = AverageMeterGroup()
        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(self.batched_train, self.batched_validate)):
            trn_X = pad_sequence([self.model.vectors[x] for x in trn_X]).permute(1,0,2)
            val_X = pad_sequence([self.model.vectors[x] for x in val_X]).permute(1,0,2)

            trn_X = trn_X.to(self.device)
            trn_y = torch.stack([y.int() for y in trn_y]).to(self.device)
            val_X = val_X.to(self.device)
            val_y = torch.stack([y.int() for y in val_y]).to(self.device)

            # phase 1. architecture step
            self.ctrl_optim.zero_grad()
            if self.unrolled:
                self._unrolled_backward(trn_X, trn_y, val_X, val_y)
            else:
                self._backward(val_X, val_y)
            self.ctrl_optim.step()

            # phase 2: child network step
            self.optimizer.zero_grad()
            logits, loss = self._logits_and_loss(trn_X, trn_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.)  # gradient clipping
            self.optimizer.step()

            metrics = self.metrics(logits, trn_y)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
                            self.num_epochs, step + 1, len(self.train_loader), meters)

    def validate_one_epoch(self, epoch):
        self.model.eval()
        self.mutator.eval()
        meters = AverageMeterGroup()
        with torch.no_grad():
            self.mutator.reset()
            for step, (X, y) in enumerate(self.batched_test):
                X = pad_sequence([self.model.vectors[x] for x in X]).permute(1,0,2)
                X = X.to(self.device)
                y = torch.stack([y_.int() for y_ in y]).to(self.device)
                logits = self.model(X)
                metrics = self.metrics(logits, y)
                meters.update(metrics)
                if self.log_frequency is not None and step % self.log_frequency == 0:
                    logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
                                self.num_epochs, step + 1, len(self.test_loader), meters)

    def _logits_and_loss(self, X, y):
        self.mutator.reset()
        logits = self.model(X)
        loss = self.loss(logits, y)
        return logits, loss

    def _backward(self, val_X, val_y):
        """
        Simple backward with gradient descent
        """
        _, loss = self._logits_and_loss(val_X, val_y)
        loss.backward()

    def _unrolled_backward(self, trn_X, trn_y, val_X, val_y):
        """
        Compute unrolled loss and backward its gradients
        """
        backup_params = copy.deepcopy(tuple(self.model.parameters()))

        # do virtual step on training data
        lr = self.optimizer.param_groups[0]["lr"]
        momentum = self.optimizer.param_groups[0]["momentum"]
        weight_decay = self.optimizer.param_groups[0]["weight_decay"]
        self._compute_virtual_model(trn_X, trn_y, lr, momentum, weight_decay)

        # calculate unrolled loss on validation data
        # keep gradients for model here for compute hessian
        _, loss = self._logits_and_loss(val_X, val_y)
        w_model, w_ctrl = tuple(self.model.parameters()), tuple(self.mutator.parameters())
        w_grads = torch.autograd.grad(loss, w_model + w_ctrl)
        d_model, d_ctrl = w_grads[:len(w_model)], w_grads[len(w_model):]

        # compute hessian and final gradients
        hessian = self._compute_hessian(backup_params, d_model, trn_X, trn_y)
        with torch.no_grad():
            for param, d, h in zip(w_ctrl, d_ctrl, hessian):
                # gradient = dalpha - lr * hessian
                param.grad = d - lr * h

        # restore weights
        self._restore_weights(backup_params)

    def _compute_virtual_model(self, X, y, lr, momentum, weight_decay):
        """
        Compute unrolled weights w`
        """
        # don't need zero_grad, using autograd to calculate gradients
        _, loss = self._logits_and_loss(X, y)
        gradients = torch.autograd.grad(loss, self.model.parameters())
        with torch.no_grad():
            for w, g in zip(self.model.parameters(), gradients):
                m = self.optimizer.state[w].get("momentum_buffer", 0.)
                w = w - lr * (momentum * m + g + weight_decay * w)

    def _restore_weights(self, backup_params):
        with torch.no_grad():
            for param, backup in zip(self.model.parameters(), backup_params):
                param.copy_(backup)

    def _compute_hessian(self, backup_params, dw, trn_X, trn_y):
        """
            dw = dw` { L_val(w`, alpha) }
            w+ = w + eps * dw
            w- = w - eps * dw
            hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
            eps = 0.01 / ||dw||
        """
        self._restore_weights(backup_params)
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        if norm < 1E-8:
            logger.warning("In computing hessian, norm is smaller than 1E-8, cause eps to be %.6f.", norm.item())

        dalphas = []
        for e in [eps, -2. * eps]:
            # w+ = w + eps*dw`, w- = w - eps*dw`
            with torch.no_grad():
                for p, d in zip(self.model.parameters(), dw):
                    p += e * d

            _, loss = self._logits_and_loss(trn_X, trn_y)
            dalphas.append(torch.autograd.grad(loss, self.mutator.parameters()))

        dalpha_pos, dalpha_neg = dalphas  # dalpha { L_trn(w+) }, # dalpha { L_trn(w-) }
        hessian = [(p - n) / 2. * eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

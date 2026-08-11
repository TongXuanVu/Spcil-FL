import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy, calculate_metrics
from scipy.spatial.distance import cdist
import os

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args.get("memory_size", 5000)
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        # Che do replay theo ty le phan tram tren tung lop (proportional).
        # Vi du memory_percent = 0.01 => moi lop giu lai 1% so mau cua chinh no.
        self._memory_percent = args.get("memory_percent", None)
        self._class_sample_counts = {}
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def _num_samples_for_class(self, data_manager, class_idx):
        """So mau train cua mot lop (cache lai de tranh load lai nhieu lan)."""
        if class_idx not in self._class_sample_counts:
            data, _, _ = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            self._class_sample_counts[class_idx] = len(data)
        return self._class_sample_counts[class_idx]

    def _m_for_class(self, data_manager, class_idx, default_m):
        """
        Tra ve so mau replay cho mot lop.
        - Neu bat che do memory_percent: giu lai round(percent * so_mau_cua_lop), toi thieu 1.
        - Nguoc lai: dung default_m (co che cu).
        """
        if self._memory_percent is not None:
            n = self._num_samples_for_class(data_manager, class_idx)
            return max(1, int(round(n * self._memory_percent)))
        return default_m

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true, loss=None):
        ret = {}
        metrics = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = metrics
        ret["top1"] = metrics["total"]
        
        ret["precision_micro"] = metrics["precision_micro"]
        ret["precision_macro"] = metrics["precision_macro"]
        ret["precision_weighted"] = metrics["precision_weighted"]
        
        ret["recall_micro"] = metrics["recall_micro"]
        ret["recall_macro"] = metrics["recall_macro"]
        ret["recall_weighted"] = metrics["recall_weighted"]
        
        ret["f1_micro"] = metrics["f1_micro"]
        ret["f1_macro"] = metrics["f1_macro"]
        ret["f1_weighted"] = metrics["f1_weighted"]
        ret["loss"] = round(loss, 6) if loss is not None else 0.0
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self, save_conf=False):
        y_pred, y_true, cnn_loss = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true, loss=cnn_loss)

        if hasattr(self, "_class_means"):
            y_pred, y_true, _ = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        if save_conf:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
            _target_path = os.path.join(self.args['logfilename'], "target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

            _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
            with open(_save_path, "a+") as f:
                f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")

        return cnn_accy, nme_accy, y_pred, y_true

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        y_pred, y_true = [], []
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return calculate_metrics(np.concatenate(y_true), np.concatenate(y_pred))

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        total_loss, num_samples = 0.0, 0
        criterion = torch.nn.CrossEntropyLoss()
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets_dev = targets.to(self._device).long()
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
                loss = criterion(outputs, targets_dev)
                total_loss += loss.item() * inputs.size(0)
                num_samples += inputs.size(0)
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        avg_loss = total_loss / max(1, num_samples)
        return np.concatenate(y_pred), np.concatenate(y_true), avg_loss  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        y_pred, y_true = [], []
        
        # Process in batches to avoid OOM on Kaggle with huge test sets (> 14M samples)
        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                inputs = _inputs.to(self._device)
                
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(self._network.module.extract_vector(inputs))
                else:
                    _vectors = tensor2numpy(self._network.extract_vector(inputs))
                
                # Normalize batch vectors [batch_size, feature_dim]
                norms = np.linalg.norm(_vectors, axis=1, keepdims=True) + EPSILON
                _vectors = _vectors / norms
                
                # Compute distance for this batch only [nb_classes, batch_size]
                dists = cdist(class_means, _vectors, "sqeuclidean")
                scores = dists.T  # [batch_size, nb_classes]
                
                # Get topk predictions
                preds = np.argsort(scores, axis=1)[:, : self.topk]
                
                y_pred.append(preds)
                y_true.append(_targets)

        if len(y_pred) == 0:
            return np.array([]), np.array([]), None
            
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

        return y_pred, y_true, None

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        if len(vectors) == 0:
            return np.array([]), np.array([])
        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            m_c = self._m_for_class(data_manager, class_idx, m)
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m_c], dummy_targets[mask][:m_c]
            
            if len(dd) == 0:
                continue

            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Fast Exemplar mean computation (Bypass DataLoader completely)
            self._network.eval()
            with torch.no_grad():
                bz = self.args.get("batch_size", 512)
                vectors_list = []
                for i in range(0, len(dd), bz):
                    batch_x = torch.tensor(dd[i:i+bz], dtype=torch.float32).to(self._device)
                    if isinstance(self._network, nn.DataParallel):
                        v = self._network.module.extract_vector(batch_x)
                    else:
                        v = self._network.extract_vector(batch_x)
                    vectors_list.append(v.cpu().numpy())
                
                vectors = np.concatenate(vectors_list, axis=0)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                mean = np.mean(vectors, axis=0)
                mean = mean / np.linalg.norm(mean)
                self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            # Fast index retrieval
            idxes = np.where(data_manager._train_targets == class_idx)[0]
            num_samples = len(idxes)
            self._class_sample_counts[class_idx] = num_samples
            
            if num_samples == 0:
                continue

            if self._memory_percent is not None:
                m_c = max(1, int(round(num_samples * self._memory_percent)))
            else:
                m_c = m
            selected_m = min(m_c, num_samples)
            
            rng = np.random.default_rng(int(self.args.get("seed", 0)) + int(class_idx))
            chosen = rng.choice(num_samples, size=selected_m, replace=False)
            
            # Extract chosen samples directly
            selected_indices = idxes[chosen]
            selected_exemplars = data_manager._train_data[selected_indices]
            exemplar_targets = np.full(selected_m, class_idx)
            
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Fast Exemplar mean computation (Bypass DataLoader completely)
            self._network.eval()
            with torch.no_grad():
                bz = self.args.get("batch_size", 512)
                vectors_list = []
                for i in range(0, selected_m, bz):
                    batch_x = torch.tensor(selected_exemplars[i:i+bz], dtype=torch.float32).to(self._device)
                    if isinstance(self._network, nn.DataParallel):
                        v = self._network.module.extract_vector(batch_x)
                    else:
                        v = self._network.extract_vector(batch_x)
                    vectors_list.append(v.cpu().numpy())
                
                vectors = np.concatenate(vectors_list, axis=0)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                mean = np.mean(vectors, axis=0)
                mean = mean / np.linalg.norm(mean)
                self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=0
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            num_samples = len(data)
            # Cache so mau cua lop de _reduce_exemplar dung lai o cac task sau.
            self._class_sample_counts[class_idx] = num_samples
            
            if num_samples == 0:
                continue

            if self._memory_percent is not None:
                m_c = max(1, int(round(num_samples * self._memory_percent)))
            else:
                m_c = m
            selected_m = min(m_c, num_samples)
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=0
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, selected_m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(selected_m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=0
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means

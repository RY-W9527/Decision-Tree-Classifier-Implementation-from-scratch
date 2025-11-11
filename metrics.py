import numpy as np
from typing import Sequence, Tuple, Dict

# Helper function
def _ravel(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).ravel()

def _divide(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0

# Function to write metrics to file
def write_metrics(f, metric_name: str, metric: dict):
    f.write(f"====== {metric_name} =====\n")
    for key, value in metric.items():
        # Matrix (sum OR avg)
        if isinstance(value, dict):
            f.write(f"{key}:\n")
            for cls, score in value.items():
                f.write(f"  {cls}: {float(score):.4f}\n")
        elif isinstance(value, np.ndarray):
            f.write(f"{key}:\n"+ np.array2string(value, precision=4) + "\n")
        elif isinstance(value, (float, int, np.floating, np.integer)): # Write scalars
            f.write(f"{key}:  {float(value):.4f}\n")
        else: # Everything else
            f.write(f"{key}: {value}\n")
    f.write("\n\n")

# Helper functions for metric calculations
class MetricOperations:
    @staticmethod
    def compute_all_metrics(y_true, y_pred):
        """ Compute all metrics and return as a dictionary, including accuracy, confusion matrix, precision, recall, and F1-score.
        Args:
            y_true (numpy.ndarray): Ground truth labels.
            y_pred (numpy.ndarray): Predicted labels.
        Returns:
            dict: A dictionary containing all computed metrics.
            the metrics computed are 
            [
                'accuracy', 
                'confusion_matrix', 
                'precision', 
                'recall', 
                'f1_score'
            ].
            """
        p,r,f = MetricOperations.precision_recall_f1(y_true, y_pred)
        metrics:dict = {
            "accuracy": MetricOperations.accuracy(y_true, y_pred),
            "confusion_matrix": MetricOperations.confusion_matrix(y_true, y_pred),
            "precision": p, "recall": r, "f1_score": f
        }
        return metrics

    @staticmethod
    def accuracy(y_true, y_pred)->float:
        acc : float = np.sum(y_true == y_pred) / len(y_true)
        return acc
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence=[1,2,3,4]) -> np.ndarray:
        """Fixed-order confusion matrix (rows=groundtruth, cols=prediction)."""
        # Flatten arrays
        y_true = _ravel(y_true); y_pred = _ravel(y_pred)
        k = len(classes)
        idx = {c: i for i, c in enumerate(classes)}
        # Initialize matrix
        mat = np.zeros((k, k), dtype=int)
        # Put values in matrix
        for t, p in zip(y_true, y_pred):
            if t in idx and p in idx:
                mat[idx[t], idx[p]] += 1
        return mat
    
    
    @staticmethod
    def precision_recall_f1(
        y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence=[1,2,3,4]
    ) -> Tuple[Dict, Dict, Dict]:
        """Per-class PRF with fixed class order; returns dicts keyed by class."""
        y_true = _ravel(y_true); y_pred = _ravel(y_pred)
        idx = {c: i for i, c in enumerate(classes)}
        tp = {c: 0 for c in classes}
        fp = {c: 0 for c in classes}
        fn = {c: 0 for c in classes}

        for t, p in zip(y_true, y_pred):
            if t in idx and p in idx:
                if t == p:
                    tp[t] += 1
                else:
                    fp[p] += 1
                    fn[t] += 1

        prec = {c: _divide(tp[c], tp[c] + fp[c]) for c in classes}
        rec  = {c: _divide(tp[c], tp[c] + fn[c]) for c in classes}
        f1   = {c: _divide(2 * prec[c] * rec[c], (prec[c] + rec[c])) for c in classes}
        return prec, rec, f1
    
class Metrics(MetricOperations):
    def __init__(self):
        self.value={}
        self.average={}
    
    def compute(self, y_true, y_pred, metric_name=None):
        """Compute a specific metric.

        Args:
            y_true (numpy.ndarray): Ground truth labels.
            y_pred (numpy.ndarray): Predicted labels.
            metric_name (str): Name of the metric to compute.

        Returns:
            The computed metric value.
        """
        if metric_name is None:
            return self.compute_all_metrics(y_true, y_pred)
        elif metric_name == "accuracy":
            return {metric_name: self.accuracy(y_true, y_pred)}
        elif metric_name == "confusion_matrix":
            return {metric_name: self.confusion_matrix(y_true, y_pred)}
        elif metric_name == "precision":
            return {metric_name: self.precision_recall_f1(y_true, y_pred)[0]}
        elif metric_name == "recall":
            return {metric_name: self.precision_recall_f1(y_true, y_pred)[1]}
        elif metric_name == "f1_score":
            return {metric_name: self.precision_recall_f1(y_true, y_pred)[2]}
        else:
            raise ValueError(f"Metric '{metric_name}' is not supported.")

    def update(self, current_metrics):
        """Stack the current metrics into the overall metrics.

        Args:
            current_metrics (dict): A dictionary containing metric names as keys and their computed values.
        """
        if current_metrics is None:
            return self.value
        for key in current_metrics.keys():
            if key in self.value.keys():
                self.value[key].append(current_metrics[key])
            else:
                self.value[key] = [current_metrics[key]]
        return self.value
    
    def get_average_metrics(self):
        """Compute the average of each metric across all stored metrics.

        Returns:
            dict: A dictionary containing the average of each metric.
        """
        avg_metrics = {}
        if not self.value:
            return avg_metrics
        for key, values in self.value.items():
            if key == "confusion_matrix":
                avg_metrics[key] = np.mean(values, axis=0)
            elif key in ["precision", "recall", "f1_score"]:
                # Average per class
                summed = {}
                counts = {}
                for class_dict in values:
                    for class_label, score in class_dict.items():
                        if class_label in summed:
                            summed[class_label] += score
                            counts[class_label] += 1
                        else:
                            summed[class_label] = score
                            counts[class_label] = 1
                avg_metrics[key] = {class_label: summed[class_label]/counts[class_label] for class_label in summed}
            else:
                avg_metrics[key] = np.mean(values)
        self.average = avg_metrics
        return avg_metrics


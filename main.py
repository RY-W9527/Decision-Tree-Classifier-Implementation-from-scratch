from Dataloader import DataLoader
from DecisionTreeModel import DecisionTreeModel
import numpy as np
from metrics import Metrics, write_metrics
import os
from typing import Dict

def nested_k_fold_cross_validation(
        dataset: np.ndarray,
        save_img_path_fmt: str,
        num_folds: int=10, # 10-fold cross-validation
    )->Dict[str,Metrics]:
    """Train and evaluate the Decision Tree models with nested k-fold cross-validation. 
    Specifically, we separate the dataset into k folds, stable the last 1 fold as the testing set, in the remaining k-1 folds, we perform k-fold cross-validation:

        1. we separate the 8 folds training set from 1 fold of validation samples in the current i-th roll of the train+val dataset;
    
        2. we then train the model using this current training set, 
        
        3. test on the testing set partitioned at the start (This is for comparing performance before and after pruning);
        
        4. prune on this current validation set;
        
        5. test again on the test set after pruning.

        6. Roll the validation set to the next fold, repeat the above for all k-1 folds.
    Args:
        dataset (np.ndarray): The entire dataset.
        num_folds (int, optional): Number of folds (k) for cross-validation. Defaults to 10.
    Returns:
        dict: {
                "val_metrics":val_metrics, 
                "test_metrics":test_metrics, 
                "test_metrics_pruned":test_metrics_pruned
            }
    """
    # Initialize metrics storage
    train_metrics = Metrics()
    val_metrics = Metrics()
    test_metrics = Metrics()
    test_metrics_pruned = Metrics()
    
    # Create folds
    indices = np.arange(len(dataset)) # obtain indices
    folds = np.array_split(indices, num_folds) # Create indices for each fold
    curr_iteration_idx = 0 # to track the current iteration number

    # Outer loop: Iterate over each fold as the test set
    for k in range(num_folds):
        curr_test_set = dataset[folds[k], :]
        
        # Inner loop: Iterate over the remaining folds for training and validation
        for i in [j for j in range(num_folds) if j != k]:
            # Step 2: Create Decision Tree models
            # create a new model for each inner fold, 
            # this will be overwritten in each inner loop 
            # as we do not need to keep the previous ones
            model: DecisionTreeModel = DecisionTreeModel()

            # Roll the train and validation indices for the current fold
            validation_idx = folds[i] # use fold index as the current validation indices
            train_idx = np.hstack([folds[j] for j in range(num_folds) if (j != i and j != k)]) # use the rest as training indices

            # Get the current training and validation sets 
            curr_train_set = dataset[train_idx, :]
            curr_val_set = dataset[validation_idx, :]

            # Train the model with the current training set
            model.decision_tree_learning(curr_train_set)
            if k==0: # Plot only the first outer loop
                model.plot(save_path=save_img_path_fmt.format(k, i, "wo"))  # OPTIONAL: View the tree structure after fitting

            # OPTIONAL: Validate the model with the current TRAIN set (Optional)
            y_pred_train = model.predict(curr_train_set) # predict on current dataset
            curr_metric = train_metrics.compute(y_true=curr_train_set[:,-1], y_pred=y_pred_train, metric_name="accuracy") # compute accuracy for current set
            curr_metric.update({'depth': model.depth})
            train_metrics.update(curr_metric) # update metrics on current set to storage
            print(f"==== Fold {curr_iteration_idx+1}/{num_folds*(num_folds-1)}, Depth: {model.depth} ====")
            print(f"Train Accuracy: {train_metrics.value['accuracy'][-1]:.4f}")
            
            # OPTIONAL: Validate the model with the current VAL set (Optional)
            y_pred_val = model.predict(curr_val_set) # predict on current dataset
            curr_metric = val_metrics.compute(y_true=curr_val_set[:,-1], y_pred=y_pred_val, metric_name="accuracy") # compute accuracy for current set
            curr_metric.update({'depth': model.depth})
            val_metrics.update(curr_metric) # update metrics on current set to storage
            print(f"Val Accuracy: {val_metrics.value['accuracy'][-1]:.4f}")

            # Validate the model with the current TEST set
            y_pred_test = model.predict(curr_test_set) # predict on current dataset
            curr_metric = test_metrics.compute(y_true=curr_test_set[:,-1], y_pred=y_pred_test) # compute all metrics for current set
            curr_metric.update({'depth': model.depth})
            test_metrics.update(curr_metric)# update metrics on current set to storage
            print(f"Test Accuracy: {test_metrics.value['accuracy'][-1]:.4f}")


            # Step 4: Pruning the tree using the validation set
            model.prune(curr_train_set, curr_val_set)
            if k==0: # Plot only the first outer loop
                model.plot(save_path=save_img_path_fmt.format(k, i,"w"))  # OPTIONAL: View the tree structure after pruning

            # Validate the model with the current TEST set after pruning
            y_pred_test_pruned = model.predict(curr_test_set) # predict on current dataset
            curr_metric = test_metrics_pruned.compute(y_true=curr_test_set[:,-1], y_pred=y_pred_test_pruned) # compute all metrics for current set
            curr_metric.update({'depth': model.depth})
            test_metrics_pruned.update(curr_metric)# update metrics on current set to storage
            print(f"Test Accuracy after Pruning: {test_metrics_pruned.value['accuracy'][-1]:.4f}")

            curr_iteration_idx += 1
    val_metrics.get_average_metrics()
    test_metrics.get_average_metrics()
    test_metrics_pruned.get_average_metrics()
    return {
        "val_metrics":val_metrics, 
        "test_metrics":test_metrics, 
        "test_metrics_pruned":test_metrics_pruned
    }

def k_fold_cross_validation(
        dataset: np.ndarray,
        save_img_path_fmt: str,
        num_folds: int=10, # 10-fold cross-validation
    )->Dict[str,Metrics]:
    """Train and evaluate the Decision Tree models with k-fold cross-validation. 
    Specifically, 
        
        1. we separate the dataset into k folds, 1 fold as the testing set, 
    
        2. and the remaining k-1 folds as the training set. 
        
        3. Then rotate the test set to the next fold; 
        
        4. repeat for all k folds.

    Args:
        dataset (np.ndarray): The entire dataset.
        save_img_path_fmt (str): The format string for saving images of the tree. It should contain two placeholders for the fold indices, and the without-pruning indicator.
        num_folds (int, optional): Number of folds (k) for cross-validation. Defaults to 10.
    Returns:
        dict: {"train_metrics":train_metrics, "test_metrics":test_metrics}
    """
    # Initialize metrics storage
    train_metrics = Metrics()
    test_metrics = Metrics()
    
    # Create folds
    indices = np.arange(len(dataset)) # obtain indices
    folds = np.array_split(indices, num_folds) # Create indices for each fold
    curr_iteration_idx = 0

    # Single loop: Iterate over each fold of the test set 
    for k in range(num_folds):
        model: DecisionTreeModel = DecisionTreeModel()
        # Roll the train and validation indices for the current fold
        curr_test_set = dataset[folds[k], :]
        train_idx = np.hstack([folds[j] for j in range(num_folds) if (j != k)]) # use the rest as training indices
        curr_train_set = dataset[train_idx[:], :]

        # Train the model with the current training set
        _, depth = model.decision_tree_learning(curr_train_set)
        model.plot(save_path=save_img_path_fmt.format(k, "wo"))  # OPTIONAL: View the tree structure after fitting

        # OPTIONAL: Validate the model with the current TRAIN set (Optional)
        y_pred_train = model.predict(curr_train_set) # predict on current dataset
        curr_metric = train_metrics.compute(y_true=curr_train_set[:,-1], y_pred=y_pred_train, metric_name="accuracy") # compute accuracy for current set
        curr_metric.update({'depth': depth})
        train_metrics.update(curr_metric) # update metrics on current set to storage
        print(f"==== Fold {curr_iteration_idx+1}/{num_folds}, Depth: {depth} ====")
        print(f"Train Accuracy: {train_metrics.value['accuracy'][-1]:.4f}")
        

        # Validate the model with the current TEST set
        y_pred_test = model.predict(curr_test_set) # predict on current dataset
        curr_metric = test_metrics.compute(y_true=curr_test_set[:,-1], y_pred=y_pred_test) # compute all metrics for current set
        curr_metric.update({'depth': depth})
        test_metrics.update(curr_metric)# update metrics on current set to storage
        print(f"Test Accuracy: {test_metrics.value['accuracy'][-1]:.4f}")

        curr_iteration_idx += 1
    train_metrics.get_average_metrics()
    test_metrics.get_average_metrics()
    return {"train_metrics":train_metrics, "test_metrics":test_metrics}


def main():
    # Output Directories
    save_dir = "results"
    section_1 = "k_fold"
    section_2 = "k_fold_nested"
    save_metric_path_sec_1 = f"{save_dir}/{section_1}/metric_results.txt"
    save_metric_path_sec_2 = f"{save_dir}/{section_2}/metric_results.txt"
    save_img_dir_sec_1 = f"{save_dir}/{section_1}/graph"
    save_img_dir_sec_2 = f"{save_dir}/{section_2}/graph"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_img_dir_sec_1}", exist_ok=True)
    os.makedirs(f"{save_img_dir_sec_2}", exist_ok=True)
    clean_dataset_dir = 'wifi_db/clean_dataset.txt'
    noisy_dataset_dir = 'wifi_db/noisy_dataset.txt'

    # Initialize Parameters
    random_seed = 0
    num_folds = 10

    ## Step 1: Load and preprocess data
    # We load train and val sets together for cross-validation, 
    clean_data_loader = DataLoader(clean_dataset_dir, shuffle=True, random_seed=random_seed)    
    clean_dataset = clean_data_loader.get_data()

    noisy_data_loader = DataLoader(noisy_dataset_dir, shuffle=True, random_seed=random_seed)    
    noisy_dataset = noisy_data_loader.get_data()
   

    ## Step 3-2: 10-fold cross-validation; Reinitialize model everytime when new fold is created
    # Evaluate model on clean dataset
    clean_dataset_metrics:Dict[str, Metrics] = k_fold_cross_validation(
        dataset=clean_dataset,
        num_folds=num_folds,
        save_img_path_fmt=save_img_dir_sec_1+"/img.outeridx_{}.clean.{}_prune.png"
    )
    # Evaluate model on noisy dataset
    noisy_dataset_metrics:Dict[str, Metrics] = k_fold_cross_validation(
        dataset=noisy_dataset,
        num_folds=num_folds,
        save_img_path_fmt=save_img_dir_sec_1+"/img.outeridx_{}.noisy.{}_prune.png"
    )
    print("======= Finished k-fold cross-validation. ========")
    print("Clean dataset average test accuracy:", clean_dataset_metrics['test_metrics'].average['accuracy'])
    print("Noisy dataset average test accuracy:", noisy_dataset_metrics['test_metrics'].average['accuracy'])
    print("==================================================")

    # Save resutls to a file
    with open(save_metric_path_sec_1, "w") as f:
        for metric_name, metric in clean_dataset_metrics.items():
            write_metrics(f, metric_name="clean_dataset_"+metric_name, metric=metric.average)
        for metric_name, metric in noisy_dataset_metrics.items():
                write_metrics(f, metric_name="noisy_dataset_"+metric_name, metric=metric.average)

    ## Step 4: 10-fold nested cross-validation; Reinitialize model everytime when new test is created; prune tree with validation set in each inner fold 
    # Evaluate model on clean dataset
    clean_dataset_metrics:Dict[str, Metrics] = nested_k_fold_cross_validation(
        dataset=clean_dataset,
        num_folds=num_folds,
        save_img_path_fmt=save_img_dir_sec_2+"/img.outeridx_{}.inneridx_{}.clean.{}_prune.png"
    )
        
    # Evaluate model on noisy dataset
    noisy_dataset_metrics:Dict[str, Metrics] = nested_k_fold_cross_validation(
        dataset=noisy_dataset,
        num_folds=num_folds,
        save_img_path_fmt=save_img_dir_sec_2+"/img.outeridx_{}.inneridx_{}.noisy.{}_prune.png"
    )
    print("==== Finished nested k-fold cross-validation. ====")
    print("Clean dataset average test pruned accuracy:", clean_dataset_metrics['test_metrics_pruned'].average['accuracy'])
    print("Noisy dataset average test pruned accuracy:", noisy_dataset_metrics['test_metrics_pruned'].average['accuracy'])
    print("==================================================")
    
    # Save results to a file
    with open(save_metric_path_sec_2, "w") as f:
        for metric_name, metric in clean_dataset_metrics.items():
            write_metrics(f, metric_name="clean_dataset_"+metric_name, metric=metric.average)
        for metric_name, metric in noisy_dataset_metrics.items():
            write_metrics(f, metric_name="noisy_dataset_"+metric_name, metric=metric.average)

if __name__ == "__main__":
    main()

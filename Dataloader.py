import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))  # folder where this file lives

class DataLoader:
    def __init__(self, file_path, shuffle=True, random_seed=0):
        self.file_path = os.path.join(base_dir, file_path)
        self.data = None
        self._rng = np.random.default_rng(random_seed) # Local random generator for reproduction
        self.__load_data(shuffle=shuffle)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __load_data(self, shuffle=True):
        """Get the raw data from the file.
        Args:
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        """
        # Code to load data from the file_path
        self.data = np.loadtxt(self.file_path)
        if shuffle:
            self.data = self.data[np.random.permutation(self.data.shape[0])]

    def get_data(self, split_ratio=None):
        """Return dataset or splits.

        Args:
            split_ratio (list|tuple|None):
                - None: return full dataset (ndarray).
                - len==3: (train, val, test) with ratios [t, v, u], remainder goes to test.
                - len==2: (train_val, test) with ratios [train_val, test].
        """
        # Return Data 
        if split_ratio is None: 
            return self.data
        
        # Create Split Set
        train_set = None
        validation_set = None
        test_set = None
        n_total = self.data.shape[0]

        if isinstance(split_ratio, (list, tuple)):
            assert (len(split_ratio) in (2,3) and all(0 <= r <= 1 for r in split_ratio)), "split_ratio elements must be in [0,1] with length 2 or 3."
            
            if len(split_ratio)==3:
                n_train = int(n_total * split_ratio[0])
                n_val = int(n_total * split_ratio[1])

                train_set = self.data[:n_train]
                validation_set = self.data[n_train:n_train + n_val]
                test_set = self.data[n_train + n_val:]

                return train_set, validation_set, test_set
            
            elif len(split_ratio)==2:
                n_train = int(n_total * split_ratio[0])

                train_set = self.data[:n_train]
                test_set = self.data[n_train:]

                return train_set, test_set
            
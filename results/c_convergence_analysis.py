# %%
import numpy as np

import datasets
import matplotlib.pyplot as plt
from data_preprocessing import create_case_control_dataset
from optimization.metrics import c_error

from optimization import JointClassifier, CccpClassifier, MMClassifier, DccpClassifier

c_values = np.arange(0.1, 1, 0.1)

classifiers = {
    'Joint': JointClassifier(),
    'CCCP': CccpClassifier(verbosity=1, tol=1e-5, max_iter=40, get_info=True),
    'MM': MMClassifier(verbosity=1, tol=1e-5, max_iter=40, get_info=True),
    'DCCP': DccpClassifier(tau=1, verbosity=1, tol=1e-5, max_iter=40, get_info=True),
}

c_values = [0.3, 0.5, 0.7]

for target_c in c_values:
    X, y = datasets.get_datasets()['wdbc']

    errors = {}
    for clf_name in classifiers:
        errors[clf_name] = []

    for i in range(10):
        X_new, y_new, s, c = create_case_control_dataset(X, y, target_c)

        from data_preprocessing import preprocess

        X_train, X_test, y_train, y_test, s_train, s_test = preprocess(X_new, y_new, s, test_size=0.2)

        for clf_name in classifiers:
            clf = classifiers[clf_name]
            clf.fit(X_train, s_train)

            c_errors = np.abs(np.array(clf.c_history) - target_c)
            errors[clf_name].append(c_errors)


    def numpy_unique_ordered(seq):
        """Remove duplicate from a list while keeping order with Numpy.unique
        Required:
            seq: A list containing all items
        Returns:
            A list with only unique values. Only the first occurence is kept.
        """
        array_unique = np.unique(seq, return_index=True)
        dstack = np.dstack(array_unique)
        dstack.dtype = np.dtype([('v', dstack.dtype), ('i', dstack.dtype)])
        dstack.sort(order='i', axis=1)
        return dstack.flatten()['v'].tolist()

    plt.figure()
    for clf_name in classifiers:
        e = np.mean(np.array(errors[clf_name]), axis=0)
        e = numpy_unique_ordered(e)
        plt.plot(e)
    plt.title(f'{target_c}')
    plt.show()
    plt.close()

# Import the libraries and the runAll function
import runAll as RA
import os

# This function gets the required data, gets the optimum parameters, and rund the SVM and MLP models
# With both scaling and no scaling, as well as the decision tree
def main():
    a,b,c,d = RA.getdata(4)

    print('2. SVM age')
    print(RA.svc_param_selection(c,d,3))
    print(RA.train_svm(a,b,c,d,1e-05, 0.0001, 'linear'))
    print('2. MLP age')
    print(RA.mlp_param_selection(c,d,2))
    print(RA.train_mlp(a,b,c,d, 'identity', 1, (50, 50, 50),  'constant', 'adam'))

    # With optimization

    print('2. SVM age optimized')
    print(RA.train_svm(a,b,c,d,0.01, 0.1, 'sigmoid'))
    print('2. MLP age optimized')
    print(RA.train_mlp(a,b,c,d, 'tanh', 0.1, (100, 100, 100),  'adaptive', 'sgd'))

    # Test decision tree accuracy

    print('2. decision tree age')
    print(RA.train_decision_tree(a, b, c, d))

    print(RA.train_svm1(a,b,c,d))

    # Produce test csv files, by printing in the runall functions
    print(RA.train_svm1(a, b, c, d, e ))
    print(RA.train_mlp(a, b, c, d, e, 'identity', 1, (50, 50, 50), 'constant', 'adam'))


if __name__ == '__main__':
    main()


# Import the libraries and the runAll function
import runAll as RA
import os

# This function gets the required data, gets the optimum parameters, and rund the SVM and MLP models
# With both scaling and no scaling, as well as the decision tree
def main():
    a,b,c,d = RA.getdata(2)
    print('4. SVM glasses')
    print(RA.svc_param_selection(c,d,3))
    print(RA.train_svm(a,b,c,d,1e-05, 0.01, 'linear'))
    print('4. MLP glasses')
    print(RA.mlp_param_selection(c,d,2))
    print(RA.train_mlp(a,b,c,d, 'identity', 0.001, (100, 100, 100),  'adaptive', 'lbfgs'))

    # With optimization

    print('4. SVM glasses optimization')
    print(RA.train_svm(a,b,c,d,0.001, 0.1, 'linear'))
    print('4. MLP glasses optimization')
    print(RA.train_mlp(a,b,c,d, 'tanh', 1, (50, 100, 50),  'constant', 'adam'))

    # Test decision tree accuracy

    print('4. decision tree glasses')
    print(RA.train_decision_tree(a, b, c, d))

    # Produce test csv files
    print(RA.train_mlp(a, b, c, d, e, 'tanh', 1, (50, 100, 50), 'constant', 'adam'))
    print(RA.train_svm(a, b, c, d, e, 1e-05, 0.0001, 'linear'))


if __name__ == '__main__':
    main()

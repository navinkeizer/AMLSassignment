# Import the libraries and the runAll function
import runAll as RA
import os

# This function gets the required data, gets the optimum parameters, and rund the SVM and MLP models
# With both scaling and no scaling, as well as the decision tree
def main():
    a,b,c,d = RA.getdata(5)
    print('3. SVM human')
    print(RA.svc_param_selection(c,d,2))
    print(RA.train_svm(a,b,c,d,0.001, 0.001, 'poly'))
    print('3. MLP human')
    print(RA.mlp_param_selection(c,d,2))
    print(RA.train_mlp(a,b,c,d, 'identity', 0.0001, (100,),  'invscaling', 'lbfgs'))

    # With optimization

    print('3. SVM human optimized')
    print(RA.train_svm(a,b,c,d,0.01, 10, 'rbf'))
    print('3. MLP human optimized')
    print(RA.train_mlp(a,b,c,d, 'relu', 1, (50, 50, 50),  'adaptive', 'lbfgs'))

    # Test decision tree accuracy

    print('3. decision tree human')
    print(RA.train_decision_tree(a, b, c, d))

    # Produce test csv files
    print(RA.train_mlp(a,b,c,d, e, 'relu', 1, (50, 50, 50),  'adaptive', 'lbfgs'))



if __name__ == '__main__':
    main()

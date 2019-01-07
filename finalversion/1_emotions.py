# Import the libraries and the runAll function
import runAll as RA
import os

# This function gets the required data, gets the optimum parameters, and rund the SVM and MLP models
# With both scaling and no scaling, as well as the decision tree
def main():
    # get the training and testing data using the feature extractor in runALl
    a,b,c,d = RA.getdata(3)


    print('1. SVM emotions')
    print(RA.svc_param_selection(c,d,2))
    print(RA.train_svm(a,b,c,d,0.001, 0.01, 'linear'))
    print('1. MLP emotions')
    print(RA.mlp_param_selection(c,d,2))
    print(RA.train_mlp(a,b,c,d, 'relu', 0.01, (100, 100, 100),  'constant', 'lbfgs'))

    # With scaling and optimization

    print('1. SVM emotions optimized')
    print(RA.train_svm(a,b,c,d,0.01, 10, 'rbf'))
    print('1. MLP emotions optimaized')
    print(RA.train_mlp(a,b,c,d, 'relu', 1, (100,),  'adaptive', 'adam'))

    # Test decision tree accuracy

    print('1. decision tree emotions')
    print(RA.train_decision_tree(a, b, c, d))

    # PLot the learning curve
    RA.plot_learningcurve_SVM(c,d)

    # Produce test csv files
    print(RA.train_mlp(a,b,c,d,e, 'relu', 1, (100,),  'adaptive', 'adam'))



if __name__ == '__main__':
    main()
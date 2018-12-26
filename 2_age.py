import runAll as RA
import os


def main():
    a,b,c,d = RA.getdata(4)

    # print('2. SVM age')
    # print(RA.svc_param_selection(c,d,3))
    # print(RA.train_svm(a,b,c,d,1e-05, 0.0001, 'linear'))
    # print('2. MLP age')
    # print(RA.mlp_param_selection(c,d,2))
    # print(RA.train_mlp(a,b,c,d, 'identity', 1, (50, 50, 50),  'constant', 'adam'))

    # With optimization

    # print('2. SVM age optimized')
    # print(RA.train_svm(a,b,c,d,0.01, 0.1, 'sigmoid'))
    # print('2. MLP age optimized')
    # print(RA.train_mlp(a,b,c,d, 'tanh', 0.1, (100, 100, 100),  'adaptive', 'sgd'))

    # Test decision tree accuracy

    print('2. decision tree age')
    print(RA.train_decision_tree(a, b, c, d))


if __name__ == '__main__':
    main()


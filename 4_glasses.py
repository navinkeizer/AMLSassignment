import runAll as RA
import os


def main():
    a,b,c,d = RA.getdata(2)
    # print('4. SVM glasses')
    # print(RA.svc_param_selection(c,d,3))
    # print(RA.train_svm(a,b,c,d,1e-05, 0.01, 'linear'))
    # print('4. MLP glasses')
    # print(RA.mlp_param_selection(c,d,2))
    # print(RA.train_mlp(a,b,c,d, 'identity', 0.001, (100, 100, 100),  'adaptive', 'lbfgs'))

    # With optimization

    # print('4. SVM glasses optimization')
    # print(RA.train_svm(a,b,c,d,0.001, 0.1, 'linear'))
    # print('4. MLP glasses optimization')
    # print(RA.train_mlp(a,b,c,d, 'tanh', 1, (50, 100, 50),  'constant', 'adam'))

    # Test decision tree accuracy

    print('4. decision tree glasses')
    print(RA.train_decision_tree(a, b, c, d))


if __name__ == '__main__':
    main()

import numpy as np
import csv
import pandas as pd
import openpyxl
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def softmax(z):
    z_exp = np.exp(z)
    z_sum = np.sum(z_exp)
    z_exp /= z_sum
    return z_exp

def load(path):
    f = open(path, 'r')
    csv_reader = csv.reader(f, delimiter='\t')
    X = []
    Y = []
    for record in csv_reader:
        line = record[0].split(',')
        line = [int(i) for i in line]
        Y.append(line[0])
        line[0] = 1
        X.append(line)


    return X,Y

def initialize(init_flag, features_num, labels_num, hidden_units):
    if init_flag == 1:
        alpha_star = np.random.uniform(-0.1, 0.1, (hidden_units, features_num))
        alpha_b = np.full((hidden_units, 1), 0)
        alpha = np.hstack((alpha_b, alpha_star))

        beta_star = np.random.uniform(-0.1, 0.1, (labels_num, hidden_units))
        beta_b = np.full((labels_num, 1), 0)
        beta = np.hstack((beta_b, beta_star))
    else:
        alpha_star = np.zeros((hidden_units, features_num))
        alpha_b = np.full((hidden_units, 1), 0)
        alpha = np.hstack((alpha_b, alpha_star))

        beta_star = np.zeros((labels_num, hidden_units))
        beta_b = np.full((labels_num, 1), 0)
        beta = np.hstack((beta_b, beta_star))

    return np.mat(alpha_star), np.mat(alpha), np.mat(beta_star), np.mat(beta)
def NNForward(x, alpha, beta):
    a = alpha.dot(x)
    z_small = sigmoid(a)
    z_1 = np.mat(np.array([1]))
    z = np.hstack((z_1, z_small))

    b = z.dot(beta.T)
    y_hat = softmax(b)
    return y_hat, z_small, z
def SGD(num_epoch, X_train, Y_train, alpha, beta, beta_star,learning_rate, X_valid, Y_valid):
    train_entropy_list = []
    valid_entropy_list = []
    for i in range(num_epoch):
        for j in range(len(X_train)):
            x = np.array(X_train[j])
            # NNForward
            # a = alpha.dot(x)
            # z_small = sigmoid(a)
            # z_1 = np.mat(np.array([1]))
            # z = np.hstack((z_1, z_small))
            #
            # b = z.dot(beta.T)
            # y_hat = softmax(b)
            y_hat, z_small, z = NNForward(x, alpha, beta)

            y = np.mat(np.zeros(10))
            y[0, Y_train[j]] = 1
            # cross_entropy = -np.sum(np.multiply(y, np.log(y_hat)))

            # NNBackward
            dloss_db = y_hat - y
            # print(dloss_db)

            dloss_dbeta = np.dot(dloss_db.T, z)
            # print(dloss_dbeta)

            dloss_dz = np.dot(dloss_db, beta_star)
            # print(dloss_dz)

            dloss_da = np.multiply(dloss_dz, np.multiply(z_small, 1 - z_small))

            dloss_dalpha = dloss_da.T.dot(np.mat(x))

            new_alpha = alpha - learning_rate * dloss_dalpha
            new_beta = beta - learning_rate * dloss_dbeta


            alpha = new_alpha
            # alpha_star = alpha[:, 1:1 + features_num]
            beta = new_beta
            beta_star = beta[:, 1:1 + hidden_units]


        entropy_train = 0
        for j in range(len(X_train)):
            y_hat,z_small, z= NNForward(X_train[j], alpha, beta)
            y = np.mat(np.zeros(10))
            y[0, Y_train[j]] = 1
            entropy_train -= np.sum(np.multiply(y, np.log(y_hat)))
        train_entropy_list.append(entropy_train/len(X_train))

        entropy_valid = 0
        for j in range(len(X_valid)):
            y_hat, z_small, z = NNForward(X_valid[j], alpha, beta)
            y = np.mat(np.zeros(10))
            y[0, Y_valid[j]] = 1
            entropy_valid -= np.sum(np.multiply(y, np.log(y_hat)))
        valid_entropy_list.append(entropy_valid / len(X_valid))
    return alpha, beta , train_entropy_list, valid_entropy_list

def gen_result_error(X, alpha, beta, Y):
    output_label = []
    total_error = 0
    for i in range(len(X)):
        y_hat = NNForward(X[i], alpha, beta)[0]
        # y_hat = np.array(y_hat)
        result = np.argmax(y_hat)
        output_label.append(result)

        if result != Y[i]:
            total_error += 1
    return output_label, total_error / len(X)

# import sys
# if __name__ == '__main__':
#     train_input = sys.argv[1]
#     validation_input = sys.argv[2]
#     train_out = sys.argv[3]
#     validation_out = sys.argv[4]
#     metrics_out = sys.argv[5]
#     num_epoch = int(sys.argv[6])
#     hidden_units = int(sys.argv[7])
#     init_flag = int(sys.argv[8])
#     learning_rate = float(sys.argv[9])
train_input = r'handout\tinyTrain.csv'
validation_input = r'handout\tinyValidation.csv'
train_out = 'train_out.labels'
validation_out = 'validation_out.labels'
metrics_out = 'metrics_out.txt'
num_epoch = 1
hidden_units = 4
init_flag = 2
learning_rate = 0.1

X_train, Y_train = load(train_input)
X_valid, Y_valid = load(validation_input)
features_num = len(X_train[0])-1
labels_num = 10
alpha_star, alpha, beta_star, beta = initialize(init_flag, features_num, labels_num, hidden_units)
alpha, beta, train_entropy_list, valid_entropy_list = SGD(num_epoch, X_train, Y_train, alpha, beta, beta_star, learning_rate, X_valid, Y_valid)

train_output, train_error = gen_result_error(X_train, alpha, beta, Y_train)
# file_train = open(train_out,'w')
# for i in range(len(train_output)):
#     file_train.write(str(train_output[i])+"\n")
# file_train.close()
#
# validation_output, valid_error = gen_result_error(X_valid, alpha, beta, Y_valid)
# file_valid = open(validation_out,'w')
# for i in range(len(validation_output)):
#     file_valid.write(str(validation_output[i])+"\n")
# file_valid.close()
#
# with open(metrics_out, "w") as f:
#     for i in range(len(train_entropy_list)):
#         f.write("epoch="+str(i+1)+" crossentropy(train): "+str(train_entropy_list[i])+"\n")
#         f.write("epoch=" + str(i + 1) + " crossentropy(validation): " + str(valid_entropy_list[i]) + "\n")
#     f.write("error(train): " + str(train_error) + "\n" + "error(test): " + str(valid_error))

print(train_entropy_list)
# df = pd.DataFrame()
# df['train_cross_entropy'] = train_entropy_list
# df['valid_cross_entropy'] = valid_entropy_list
# df.to_excel('plot3.xlsx')
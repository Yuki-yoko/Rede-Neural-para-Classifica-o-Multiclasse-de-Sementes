import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    r"C:\Users\yukic\OneDrive\Documentos\archive\seeds_dataset.txt",
    sep="\s+",
    header=None
)

df.columns = [
    "area","perimeter","compactness",
    "length","width","asymmetry","groove","class"
]

# embaralhar dataset
df = df.sample(frac=1).reset_index(drop=True)


X_data = df.drop(['class'], axis=1).values
classes = df['class']

# normalização
X_data = (X_data - np.mean(X_data, axis=0)) / np.std(X_data, axis=0)


Y_data = np.zeros((210,3))

for i in range(210):

    if classes[i] == 1:
        Y_data[i,0] = 1

    elif classes[i] == 2:
        Y_data[i,1] = 1

    elif classes[i] == 3:
        Y_data[i,2] = 1

X_data = np.append(X_data, np.ones((210,1)), axis=1)


X_train = X_data[:190]
Y_train = Y_data[:190]

X_test = X_data[190:]
Y_test = Y_data[190:]


def activation(x):
    return 1/(1+np.exp(-x))


NN = np.random.rand(3,8) - 0.5

# treinamento
learning_rate = 0.005
epochs = 10000


loss_vector = []

for epoch in range(epochs):

    epoch_loss = 0

    for idx in range(X_train.shape[0]):

        input_instance = X_train[idx]

        output = activation(np.matmul(input_instance, NN.T))

        error = output - Y_train[idx]

        epoch_loss += np.sum(error**2)

        # derivada da sigmoid
        delta = error * output * (1-output)

        for i in range(3):
            for j in range(8):

                NN[i,j] -= learning_rate * delta[i] * input_instance[j]

    loss_vector.append(epoch_loss)

# grafico loss
plt.figure(dpi=300)

plt.plot(loss_vector)

plt.title("Curva de Loss")
plt.xlabel("Épocas")
plt.ylabel("Erro")

plt.grid()
plt.show()

# teste
test_outputs = activation(np.matmul(X_test, NN.T))
test_errors = test_outputs - Y_test

MSE = 0

for i in range(test_errors.shape[0]):
    for j in range(test_errors.shape[1]):
        MSE += test_errors[i,j]**2

MSE /= (test_errors.shape[0] * test_errors.shape[1])

# matriz de confusao

CM = np.zeros((3,3))

for instance in range(X_test.shape[0]):

    input_instance = X_test[instance]
    output = activation(np.matmul(input_instance, NN.T))

    prediction_class = np.argmax(output)

    pred_vector = np.zeros(3)
    pred_vector[prediction_class] = 1

    real_vector = Y_test[instance]

    print("Real:", real_vector, "Predito:", pred_vector)

    real_class = np.argmax(real_vector)

    CM[real_class, prediction_class] += 1

print("\nMatriz de Confusão:")
print(CM)

# acurácia global
accuracy = np.trace(CM) / np.sum(CM)
print("\nAccuracy:", accuracy)

# precisão, recall e F1
precision = []
recall = []
f1 = []

for i in range(3):
    TP = CM[i,i]
    FP = np.sum(CM[:,i]) - TP
    FN = np.sum(CM[i,:]) - TP

    if TP+FP == 0:
        prec = 0
    else:
        prec = TP/(TP+FP)

    if TP+FN == 0:
        rec = 0
    else:
        rec = TP/(TP+FN)

    if prec+rec == 0:
        f1_score = 0
    else:
        f1_score = 2*(prec*rec)/(prec+rec)

    precision.append(prec)
    recall.append(rec)
    f1.append(f1_score)

print("\nMSE:", MSE)
print("\nMétricas por classe:\n")

for i in range(3):
    print(f"Classe {i+1}:")
    print(f"  Precision: {float(precision[i]):.3f}")
    print(f"  Recall:    {float(recall[i]):.3f}")
    print(f"  F1-score:  {float(f1[i]):.3f}")
    print()

# gráfico predito vs real

real_classes = []
pred_classes = []

for instance in range(X_test.shape[0]):

    input_instance = X_test[instance]
    output = activation(np.matmul(input_instance, NN.T))

    pred_class = np.argmax(output)
    real_class = np.argmax(Y_test[instance])

    real_classes.append(real_class)
    pred_classes.append(pred_class)


plt.figure(dpi=300)

x = np.arange(len(real_classes))

plt.plot(x, real_classes, label="Real")
plt.plot(x, pred_classes, label="Predito")
plt.xlabel("Amostras de teste")
plt.ylabel("Classe")
plt.title("Real vs Predito")
plt.legend()
plt.grid()
plt.show()
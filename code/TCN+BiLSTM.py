import numpy as np
import scipy.io as spio
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
import argparse
from imblearn.over_sampling import SMOTE
import torch.optim as optim
from torch.utils.data import TensorDataset
from torchsummary import summary
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import random
import matplotlib.pyplot as plt
import seaborn as sns

import torch
torch.cuda.empty_cache()

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
def read_mitbih(filename, max_time=100, classes= ['F', 'N', 'S', 'V', 'Q'], max_nlabel=100):
    def normalize(data):
        data = np.nan_to_num(data)  # removing NaNs and Infs
        data = data - np.mean(data)
        data = data / np.std(data)
        return data

    def butter_lowpass_filter(data, cutoff_freq, fs, order=5):
        nyquist_freq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    # read data
    data = []
    samples = spio.loadmat(filename + ".mat")
    samples = samples['s2s_mitbih']
    values = samples[0]['seg_values']
    labels = samples[0]['seg_labels']
    num_annots = sum([item.shape[0] for item in values])

    n_seqs = num_annots / max_time
    l_data = 0
    for i, item in enumerate(values):
        l = item.shape[0]
        for itm in item:
            if l_data == n_seqs * max_time:
                break
            data.append(itm[0])
            l_data = l_data + 1

    l_lables  = 0
    t_lables = []
    for i, item in enumerate(labels):
        if len(t_lables) == n_seqs*max_time:
            break
        item= item[0]
        for lebel in item:
            if l_lables == n_seqs * max_time:
                break
            t_lables.append(str(lebel))
            l_lables = l_lables + 1

    del values
    data = np.asarray(data)
    shape_v = data.shape
    data = np.reshape(data, [shape_v[0], -1])
    t_lables = np.array(t_lables)
    _data  = np.asarray([],dtype=np.float64).reshape(0,shape_v[1])
    _labels = np.asarray([],dtype=np.dtype('|S1')).reshape(0,)
    for cl in classes:
        _label = np.where(t_lables == cl)
        permute = np.random.permutation(len(_label[0]))
        _label = _label[0][permute[:max_nlabel]]
        _data = np.concatenate((_data, data[_label]))
        _labels = np.concatenate((_labels, t_lables[_label]))

    data = _data[:(int(len(_data)/ max_time) * max_time), :]
    _labels = _labels[:(int(len(_data) / max_time) * max_time)]
    data = [data[i:i + max_time] for i in range(0, len(data), max_time)]
    labels = [_labels[i:i + max_time] for i in range(0, len(_labels), max_time)]

    permute = np.random.permutation(len(labels))
    data = np.asarray(data)
    labels = np.asarray(labels)
    # 重新验证形状是否正确
    print("Shape of data:", data.shape)
    print("Shape of labels:", labels.shape)
    data= data[permute]
    labels = labels[permute]

    data = normalize(data)
    fs = 360
    cutoff = 50
    data_filtered = np.zeros_like(data)
    for i in range(len(data)):
        data_filtered[i] = butter_lowpass_filter(data[i], cutoff, fs)
    print('Records processed!')
    return data_filtered, labels

class TCNLayer(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, dilation, padding):
        super(TCNLayer, self).__init__()
        self.conv = weight_norm(nn.Conv1d(input_size, output_size, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, dilation, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(input_size, output_size, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.relu = nn.ReLU()
        self.conv2 = weight_norm(nn.Conv1d(output_size, output_size, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.downsample = nn.Conv1d(input_size, output_size, 1) if input_size != output_size else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)
class TCNAutoencoder(nn.Module):
    def __init__(self, input_size, input_channels, num_channels, kernel_size, stride, dilation):
        super(TCNAutoencoder, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.encoder_layers = []
        self.decoder_layers = []

        num_levels = len(num_channels)
        in_channels = input_channels

        # 编码器部分
        for i in range(num_levels):
            dilation_size = dilation ** i
            padding = (kernel_size - 1) * dilation_size // 2
            out_channels = num_channels[i] if i != num_levels - 1 else input_channels
            self.encoder_layers.append(TCNLayer(in_channels, out_channels, kernel_size, stride, dilation_size, padding))
            # self.encoder_layers.append(
            #     ResidualBlock(in_channels, out_channels, kernel_size, stride, dilation_size, padding))
            in_channels = out_channels
        self.encoder = nn.Sequential(*self.encoder_layers)

        # 解码器部分
        for i in range(num_levels - 1, -1, -1):  #倒序迭代到第一层，步长为向前推进1
            dilation_size = dilation ** i
            padding = (kernel_size - 1) * dilation_size // 2
            out_channels = num_channels[i] if i != 0 else input_channels
            self.decoder_layers.append(TCNLayer(in_channels, out_channels, kernel_size, stride, dilation_size, padding))
            # self.decoder_layers.append(
            #     ResidualBlock(in_channels, out_channels, kernel_size, stride, dilation_size, padding))
            in_channels = out_channels  # 上一层输出等于下一层输入
        self.decoder = nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=4):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

def run_program(args):
    print(args)
    max_time = args.max_time  # 5 3 second best 10# 40 # 100
    n_oversampling = args.n_oversampling
    classes = args.classes
    filename = args.data_dir

    def determine_label_by_priority(labels, priority_order=['F', 'S', 'V', 'N']):
        final_labels = []
        for label_vector in labels:
            sorted_labels = sorted(label_vector, key=lambda x: priority_order.index(x))
            final_labels.append(sorted_labels[0])
        return final_labels


    X, Y = read_mitbih(filename, max_time, classes=classes, max_nlabel=100000)  # 11000
    print("# of sequences: ", len(X))


    classes = np.unique(Y)
    char2numY = dict(zip(classes, range(len(classes))))
    n_classes = len(classes)
    print('Classes: ', classes)
    print('n_classes: ', n_classes) #4
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        print(cl, len(np.where(Y.flatten() == cl)[0]))

    char2numY['<GO>'] = len(char2numY)
    num2charY = dict(zip(char2numY.values(), char2numY.keys()))

    Y = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in Y]
    Y = np.array(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train = np.reshape(X_train, [X_train.shape[0] * X_train.shape[1], -1])
    y_train = y_train[:, 1:].flatten()
    nums = []
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        nums.append(len(np.where(y_train.flatten() == ind)[0]))
    ratio = {0: n_oversampling, 1: nums[1], 2: n_oversampling, 3: n_oversampling}
    sm = SMOTE(random_state=12, sampling_strategy=ratio)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    X_train = X_train[:(X_train.shape[0] // max_time) * max_time, :]
    y_train = y_train[:(X_train.shape[0] // max_time) * max_time]

    X_train = np.reshape(X_train, [-1, X_test.shape[1], X_test.shape[2]])
    y_train = np.reshape(y_train, [-1, y_test.shape[1] - 1, ])
    y_train = [[char2numY['<GO>']] + [y_ for y_ in date] for date in y_train]
    y_train = np.array(y_train)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    y_train = y_train[:, 1:]
    y_test = y_test[:, 1:]

    def train_tcn_model(X_train, X_test, args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 定义 TCN 模型
        input_size = X_train.shape[2]  # 280
        print(X_train.shape[1])  # 10
        print(X_train.shape[2])  # 280
        tcn_model = TCNAutoencoder(input_size=input_size, input_channels=X_train.shape[1],
                                   num_channels=[32, 64, 128],
                                   kernel_size=3, stride=1, dilation=2).to(device)


        summary(tcn_model, input_size=(10, input_size), device=device.type)

        batch_size = args.batch_size
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(X_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(tcn_model.parameters(), lr=args.learning_rate)
        epochs = args.epochs  # 100

        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = tcn_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {total_loss / len(train_loader):.4f}')

        with torch.no_grad():
            X_train_features = tcn_model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
            X_test_features = tcn_model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        return X_train_features, X_test_features

    X_train_features, X_test_features = train_tcn_model(X_train, X_test, args)


    def train_bilstm_model(X_train, X_test, y_train, y_test):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_size = X_train.shape[2]
        hidden_size = args.hidden_size  # 128
        num_layers = args.num_layers  # 2
        num_classes = len(np.unique(y_train))
        epochs = args.epochs  # 100
        batch_size = args.batch_size  # 100
        learning_rate = args.learning_rate  # 0.001
        assert X_train.shape[0] == y_train.shape[0], "Size mismatch between X_train and y_train"

        bilstm_model = BiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
        print(bilstm_model)
        print("Total number of parameters: ", sum(p.numel() for p in bilstm_model.parameters()))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(bilstm_model.parameters(), lr=learning_rate)

        X_train_tensor = torch.tensor(X_train)
        X_test_tensor = torch.tensor(X_test)

        train_dataset = DataLoader(TensorDataset(X_train_tensor, y_train), batch_size=batch_size, shuffle=True)
        test_dataset = DataLoader(TensorDataset(X_test_tensor, y_test), batch_size=batch_size, shuffle=False)

        total_step = len(train_dataset)
        for epoch in range(epochs):
            for i, (features, labels) in enumerate(train_dataset):
                features = features.to(device)
                labels = labels.to(device)
                labels = labels.long()
                outputs = bilstm_model(features)
                loss = criterion(outputs.transpose(1, 2), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

        y_true = []
        y_pred = []

        with torch.no_grad():
            correct = 0
            total = 0
            for features, labels in test_dataset:
                features = features.to(device)
                labels = labels.to(device)
                outputs = bilstm_model(features)
                _, predicted = torch.max(outputs.data, 2)
                total += labels.numel()

                y_true.extend(labels.cpu().numpy().flatten())
                y_pred.extend(predicted.cpu().numpy().flatten())
                precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
                correct += (predicted == labels).sum().item()


            class_report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], target_names=['F', 'N', 'S', 'V'], digits=4)
            print('Classification Report:\n', class_report)

            print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
            conf_matrix = confusion_matrix(y_true, y_pred)
            conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1, keepdims=True) * 100
            print('Confusion Matrix:\n', conf_matrix)
            print('Precision: {:.4f}'.format(precision))
            print('Recall: {:.4f}'.format(recall))
            print('F1 Score: {:.4f}'.format(f1_score))

            # 计算每个类别的特异性
            specificity = []
            for i in range(conf_matrix.shape[0]):
                tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
                fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
                specificity.append(tn / (tn + fp))

            for i, spec in enumerate(specificity):
                print(f'Specificity for class {i} ({["F", "N", "S", "V"][i]}): {spec:.4f}')

            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=['F', 'N', 'S', 'V'],
                        yticklabels=['F', 'N', 'S', 'V'])
            plt.xlabel('Predicted Values')
            plt.ylabel('Actual Values')
            plt.title('Confusion Matrix Percentage')
            plt.show()


    train_bilstm_model(X_train_features, X_test_features, y_train, y_test)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--max_time', type=int, default=10)
    parser.add_argument('--test_steps', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--data_dir', type=str, default='data/mitbih_1')
    parser.add_argument('--n_oversampling', type=int, default=12600)
    parser.add_argument('--classes', nargs='+', type=chr,
                        default=['F', 'N', 'S', 'V'])
    args = parser.parse_args()
    run_program(args)

if __name__ == '__main__':
    main()
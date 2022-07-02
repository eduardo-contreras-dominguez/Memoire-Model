import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.utils import to_categorical
from xbbg import blp

from config import MODEL_INDICATORS


class HistoricalMacroDataRetriever:
    def __init__(self, start_date="03/30/2006", start_date_returns="02/25/2006", end_date="06/14/2022",
                 data=MODEL_INDICATORS):
        self.start_date = start_date
        self.returns_start_date = start_date_returns
        self.end_date = end_date
        self.indicators = data
        self.mocked = True

    def retrieving_data(self):
        if not self.mocked:
            cols = [
                "Avg_Weekly_Hours",
                "Consumer_Expectations",
                "New_Orders",
                "LEI",
                "SP_Monthly_Rets",
                "IR_Spread",
                "NonFarm_Payrolls",
                "Financial_Conditions",
                "Industrial_Production",
                "CPI_YoY",
                # "GDP_QoQ"
            ]
            df_data = pd.DataFrame(columns=cols)
            data = {}
            for i, ticker in enumerate(self.indicators):
                if i == 0:
                    available_dates = blp.bdh(ticker, "PX_LAST", self.start_date, self.end_date).index.tolist()
                if cols[i] in ['IR_Spread', 'Credit_Index']:
                    data[cols[i]] = [float(x) for x in
                                     blp.bdh(ticker, "PX_LAST", self.start_date, self.end_date, Per='M').values]
                elif cols[i] == 'SP_Monthly_Rets':
                    L = [float(x) for x in
                         blp.bdh(ticker, "PX_LAST", self.returns_start_date, self.end_date, Per='M').values]
                    data[cols[i]] = [100 * (b - a) / a for a, b in zip(L[::1], L[1::1])]
                else:
                    data[cols[i]] = [float(x) for x in
                                     blp.bdh(ticker, "PX_LAST", self.start_date, self.end_date).values]
                    if len(data[cols[i]]) < len(available_dates):
                        available_dates = blp.bdh(ticker, "PX_LAST", self.start_date, self.end_date).index.tolist()

            for col in data.keys():
                if col != "SP_Monthly_Rets":
                    data[col] = data[col][:len(available_dates)]
                else:
                    data[col] = data[col][:len(available_dates)]

            df_target = pd.read_excel('Cycle_Clock.xlsx', index_col=0, sheet_name=1, usecols='A:B')
            mask = (df_target.index > self.returns_start_date) & (df_target.index <= self.end_date)
            df_target = df_target[mask]
            df_target = df_target.iloc[:len(available_dates), :]
            L_target = [int(x) for x in df_target.values]
            df = pd.DataFrame(data=data, columns=cols, index=available_dates)
            df['Target'] = [el - 1 for el in L_target]
            df.to_excel("Data_Training.xlsx", sheet_name='Data', engine='xlsxwriter')
        else:
            df = pd.read_excel("Data_Training.xlsx")

        return df


class EconomicModel:
    def __init__(self):
        d = HistoricalMacroDataRetriever()
        self.historical_df = d.retrieving_data().set_index(d.retrieving_data().columns[0])
        self.model_type = "SVC"
        # Machine Learning Parameters:
        self.feature = None
        # self.feature_test = None
        self.label = None
        # self.label_test = None
        self.model = None

    def ModelPreProcessing(self):
        # Scaling Features
        if self.model_type == "NN":
            scaler = StandardScaler()
            self.feature = scaler.fit_transform(self.historical_df.drop('Target', axis=1))

            # Transforming integer labels into 0,1 arrays
            self.label = to_categorical(self.historical_df['Target'], num_classes=4)
            # self.feature_train, self.feature_test, self.label_train, self.label_test = train_test_split(X, y, test_size=0.2,
            #                                                                                            random_state=5)
        else:
            self.historical_df["Target"] = self.historical_df["Target"] + 1
            self.feature = self.historical_df.drop("Target", axis=1)
            self.label = self.historical_df["Target"]

        return 0

    def Fitting_Model_NN(self):
        input_layer_size = self.feature.shape[1]
        hidden_layer_size = input_layer_size * 2
        output_layer_size = len(self.historical_df['Target'].unique())
        model = tf.keras.Sequential([
            # tf.keras.layers.Dense(input_layer_size, activation='relu'),
            # tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
            tf.keras.layers.Dense(1024, input_dim=input_layer_size, activation='relu'),
            tf.keras.layers.Dense(output_layer_size, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(self.feature, self.label, validation_split=0.15, epochs=100, batch_size=5)
        self.model = model
        return self.model

    def Fitting_Model_SVC(self):
        # Step 1: Split Dataset into Features and Label

        # Step 2: Split X and y into Train and Test Dataset.
        X_train, X_test, y_train, y_test = train_test_split(self.feature, self.label, test_size=0.25, random_state=0)

        # Step 3: Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Step 4: Fit the SVM model
        classifier = SVC(kernel='rbf', random_state=0)
        classifier.fit(X_train, y_train)
        self.model = classifier
        # Step 5: Predict the test results
        y_pred = classifier.predict(X_test)

        # Step 6: Plot the confusion matrix: Which predictions are right and which are wrong?
        # The confusion matrix will be plotted on the method: Plot_Model()
        return y_test, y_pred

    def Plot_Model(self):
        rcParams['figure.figsize'] = (18, 8)
        if self.model_type == "NN":
            plt.plot(self.model.history.history["accuracy"])
            plt.plot(self.model.history.history["val_accuracy"])
            plt.title("Model Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()
        else:
            y_test, y_pred = self.Fitting_Model_SVC()
            cf_matrix = confusion_matrix(y_test, y_pred)
            print(cf_matrix)
            sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
                        fmt='.2%', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.xticks(range(len(cf_matrix)), [i for i in range(1, 5)])
            plt.yticks(range(len(cf_matrix)), [i for i in range(1, 5)])
            plt.ylabel("Actual")
            plt.show()
        return 0


m_model = EconomicModel()
data = m_model.ModelPreProcessing()
model = m_model.Fitting_Model_SVC()
m_model.Plot_Model()

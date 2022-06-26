import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
        # Machine Learning Parameters:
        self.feature_train = None
        self.feature_test = None
        self.label_train = None
        self.label_test = None
        self.model = None

    def ModelPreProcessing(self):
        # Scaling Features
        scaler = StandardScaler()
        X = scaler.fit_transform(self.historical_df.drop('Target', axis=1))

        # Transforming integer labels into 0,1 arrays
        y = to_categorical(self.historical_df['Target'], num_classes=4)
        self.feature_train, self.feature_test, self.label_train, self.label_test = train_test_split(X, y, test_size=0.2,
                                                                                                    random_state=5)
        # Scaling Features

        # self.feature_train = scaler.fit_transform(self.feature_train)
        # self.feature_test = scaler.fit_transform(self.feature_test)

        return 0

    def Fitting_Model(self):
        # input_layer_size = self.historical_df.shape[1]
        # hidden_layer_size = input_layer_size * 2
        output_layer_size = len(self.historical_df['Target'].unique())
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(output_layer_size, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(self.feature_train, self.label_train, epochs=100)
        self.model = model
        return self.model

    def Evaluate_Model(self):
        rcParams['figure.figsize'] = (18, 8)
        plt.plot(np.arange(1, 101), self.model.history.history['loss'], label="Loss")
        plt.show()
        pass

    def Model_Prediction(self):
        predictions = self.model.predict(self.feature_test)
        expected = self.label_test
        print(expected)
        print(predictions)
        pass


m_model = EconomicModel()
data = m_model.ModelPreProcessing()
model = m_model.Fitting_Model()
m_model.Evaluate_Model()
m_model.Model_Prediction()

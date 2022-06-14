import pandas as pd
from xbbg import blp
from config import MODEL_INDICATORS
import datetime as dt
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


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
        self.model = None

    def ModelPreProcessing(self):
        X = self.historical_df.drop('Target', axis=1)
        y = self.historical_df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def Modeling(self, X, y):
        # input_layer_size = self.historical_df.shape[1]
        # hidden_layer_size = input_layer_size * 2
        output_layer_size = len(self.historical_df['Target'].unique())
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(output_layer_size, activation='sigmoid')
        ])
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(X, y, epochs=100)
        self.model = model
        return self.model

    def ModelEvaluation(self):
        rcParams['figure.figsize'] = (18, 8)
        plt.plot(np.arange(1, 101), self.model.history.history['loss'], label="Loss")
        plt.show()
        pass

    def ModelPrediction(self, X, y):
        #predictions = self
        pass

m_model = EconomicModel()
data = m_model.ModelPreProcessing()
model = m_model.Modeling(data[0], data[2])
m_model.ModelEvaluation()

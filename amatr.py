# Copyright 2020 QuantRocket LLC - All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from moonshot import Moonshot
from moonshot.commission import PerShareCommission
from moonshot.commission import PercentageCommission
from quantrocket.master import get_securities

import pandas as pd, numpy as np
import math
from collections import defaultdict



def get_df_ama(df, alpha_min, alpha_max, beta, gamma):
    alphas = defaultdict(int)
    amas = defaultdict(int)
    dposs = defaultdict(int)
    dnegs = defaultdict(int)
    ps = defaultdict(int)
    ama_serieses = [[0 for _ in df.columns]]
    h_serieses = [[0 for _ in df.columns]]
    l_serieses = [[0 for _ in df.columns]]

    # init
    for i, p in enumerate(df.head(1).values[0]):
        amas[df.columns[i]] = p
        ama_serieses[0][i] = p
        alphas[df.columns[i]] = alpha_min

    cnt = 0
    is_first_row = True
    for r in df.iterrows():
        if is_first_row:
            is_first_row = False
            for i, p in enumerate(r[1]):
                ps[df.columns[i]] = p
            continue

        cnt += 1
        row_ama = [0 for _ in range(len(r[1]))]
        row_h = [0 for _ in range(len(r[1]))]
        row_l = [0 for _ in range(len(r[1]))]
        # for each sid (columns are sids)
        for i, p in enumerate(r[1]):
            sid = df.columns[i]
            ama = alphas[sid] * p + (1-alphas[sid]) * amas[sid]
            row_ama[i] = ama
            change = (p - ps[sid]) / ps[sid]
            dpos = alphas[sid] * max(change, 0) + (1-alphas[sid]) * dposs[sid]
            dneg = -alphas[sid] * min(change, 0) + (1-alphas[sid]) * dnegs[sid]

            # depond on above
            h = (1 + beta * dnegs[sid]) * ama
            l = (1 - beta * dnegs[sid]) * ama
            row_h[i] = h
            row_l[i] = l
            pa = (p - amas[sid]) / amas[sid]
            if p > h:
                s = (beta * dpos)
                if s == 0:
                    snr = 0
                else:
                    snr = pa / (beta * dpos)
            elif p < l:
                snr = -pa / (beta * dneg)
            else:
                snr = 0

            # depend on above
            alpha = alpha_min + (alpha_max - alpha_min) * math.atan(gamma * snr) / (math.pi / 2)

            #print(f'sid: {sid}, ama: {ama}, dpos: {dpos}, dneg: {dneg}, alpha: {alpha}, h: {h}, l: {l}, snr: {snr}, p: {p}, change: {change}')
            # update
            amas[df.columns[i]] = ama
            dposs[df.columns[i]] = dpos
            dnegs[df.columns[i]] = dneg
            alphas[df.columns[i]] = alpha
            ps[df.columns[i]] = p

        ama_serieses.append(row_ama)
        h_serieses.append(row_h)
        l_serieses.append(row_l)

    df_ama = pd.DataFrame(data=ama_serieses, columns=df.columns, index=df.index)
    df_h = pd.DataFrame(data=h_serieses, columns=df.columns, index=df.index)
    df_l = pd.DataFrame(data=l_serieses, columns=df.columns, index=df.index)
    
    return df_ama, df_h, df_l


sid_snp500 = "FIBBG000BDTBL9" # "FIBBG003MVLMY1"


class AMATR(Moonshot):
    CODE = "amatr"
    DB = ["usstock-1d"]
    
    ALPHA_MIN = 0.4
    ALPHA_MAX = 0.8
    BETA = 0.5
    GAMMA = 0.5
    
    REBALANCE_INTERVAL = "W" # M = monthly; see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

    def prices_to_signals(self, prices):
        """
        This method receives a DataFrame of prices and should return a
        DataFrame of integer signals, where 1=long, -1=short, and 0=cash.
        """
        df_close = prices.loc["Close"]
        print(f"df_close:\n{df_close}")
        
        df_ama, df_h, df_l = get_df_ama(df_close, self.ALPHA_MIN, self.ALPHA_MAX, self.BETA, self.GAMMA)

        longs = (df_close >= df_h)
        shorts = (df_close <= df_l)

        longs = longs.astype(int)
        shorts = -shorts.astype(int)
        
        # Combine long and short signals
        signals = longs.where(longs == 1, shorts)

        # Resample using the rebalancing interval.
        # Keep only the last signal of the month, then fill it forward
        signals = signals.resample(self.REBALANCE_INTERVAL).last()
        signals = signals.reindex(df_close.index, method="ffill")

        return signals
    
    def signals_to_target_weights(self, signals, prices):
        """
        This method receives a DataFrame of integer signals (-1, 0, 1) and
        should return a DataFrame indicating how much capital to allocate to
        the signals, expressed as a percentage of the total capital allocated
        to the strategy (for example, -0.25, 0, 0.1 to indicate 25% short,
        cash, 10% long).
        """
        weights = self.allocate_equal_weights(signals)
        print(f"weights: \n{weights.tail(10)}")
        return weights

    def target_weights_to_positions(self, weights, prices):
        """
        This method receives a DataFrame of allocations and should return a
        DataFrame of positions. This allows for modeling the delay between
        when the signal occurs and when the position is entered, and can also
        be used to model non-fills.
        """
        # Enter the position in the period/day after the signal
        print(f"weights.shift(): \n{weights.shift().tail(10)}")
        return weights.shift()

    def positions_to_gross_returns(self, positions, prices):
        """
        This method receives a DataFrame of positions and a DataFrame of
        prices, and should return a DataFrame of percentage returns before
        commissions and slippage.
        """
        # Our return is the security's close-to-close return, multiplied by
        # the size of our position. We must shift the positions DataFrame because
        # we don't have a return until the period after we open the position
        df_close = prices.loc["Close"]
        columns_fx = [c for c in df_close.columns if 'FX' in c]
        df_fx = df_close[columns_fx]
        df_fx = df_close[positions.columns]
        print(f"positions.columns: \n{positions.columns}")
        print(f"df_fx.columns: \n{df_fx.columns}")
        print(f"positions: \n{positions.tail(10)}")
        gross_returns = df_fx.pct_change() * positions.shift()
        print(f"gross_returns: \n{gross_returns.tail(10)}")
        return gross_returns

class USStockCommission(PercentageCommission):
    BROKER_COMMISSION_RATE = 0.0005 # 0.05% of trade value

class AMATRDemo(AMATR):

    CODE = "amatr-demo"
    DB = ["usstock-1d"]
    
    COMMISSION_CLASS = USStockCommission




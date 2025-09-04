import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
import joblib
from utils import price_dict_to_df
import talib as ta
plt.style.use('dark_background')



# --- Load data ---
datas = joblib.load('/Users/leoinv/Documents/CODE/BACKTEST/data/1h.joblib')
price_df = price_dict_to_df(datas)[['BTCUSDT', 'SHIBUSDT']].dropna()
price_df.columns = ['Asset1', 'Asset2']
price_df = price_df[-10000:]

price_df['rsi1'] = ta.RSI(price_df['Asset1'].values, 40)
price_df['rsi2'] = ta.RSI(price_df['Asset2'].values, 40)

rsi = price_df[['rsi1','rsi2']].dropna()

# --- Copula plots (as before) ---
u = rsi.rank(method='average') / (len(rsi) + 1.0)
u1, u2 = u['rsi1'].values, u['rsi2'].values
values_copula = np.vstack([u1, u2])
kde_copula = gaussian_kde(values_copula, bw_method=0.05)

grid_n = 100
x = np.linspace(0,1,grid_n)
y = np.linspace(0,1,grid_n)
X, Y = np.meshgrid(x, y)
positions = np.vstack([X.ravel(), Y.ravel()])
Z_copula = np.reshape(kde_copula(positions).T, X.shape)

# --- Normal RSI plots ---
values_rsi = np.vstack([rsi['rsi1'].values, rsi['rsi2'].values])
kde_rsi = gaussian_kde(values_rsi, bw_method=0.1)

# grid for normal RSI values
x_rsi = np.linspace(rsi['rsi1'].min(), rsi['rsi1'].max(), grid_n)
y_rsi = np.linspace(rsi['rsi2'].min(), rsi['rsi2'].max(), grid_n)
X_rsi, Y_rsi = np.meshgrid(x_rsi, y_rsi)
positions_rsi = np.vstack([X_rsi.ravel(), Y_rsi.ravel()])
Z_rsi = np.reshape(kde_rsi(positions_rsi).T, X_rsi.shape)

# # --- Plotting ---
# fig = plt.figure(figsize=(16,12))

# # Copula 2D heatmap
# ax1 = fig.add_subplot(2,2,1)
# im1 = ax1.imshow(Z_copula, origin='lower', extent=[0,1,0,1],
#                  cmap="viridis", aspect="auto")
# fig.colorbar(im1, ax=ax1, label="Density")
# ax1.set_xlabel("u1 (ECDF)")
# ax1.set_ylabel("u2 (ECDF)")
# ax1.set_title("Empirical Copula (KDE heatmap)")

# # Copula 3D surface
# ax2 = fig.add_subplot(2,2,2, projection='3d')
# ax2.plot_surface(X, Y, Z_copula, cmap="viridis", edgecolor="none", alpha=0.9)
# ax2.set_xlabel("u1 (ECDF)")
# ax2.set_ylabel("u2 (ECDF)")
# ax2.set_zlabel("Density")
# ax2.set_title("Empirical Copula (KDE 3D surface)")

# # Normal RSI 2D heatmap
# ax3 = fig.add_subplot(2,2,3)
# im2 = ax3.imshow(Z_rsi, origin='lower', extent=[x_rsi.min(), x_rsi.max(),
#                                                 y_rsi.min(), y_rsi.max()],
#                  cmap="plasma", aspect="auto")
# fig.colorbar(im2, ax=ax3, label="Density")
# ax3.set_xlabel("RSI Asset1")
# ax3.set_ylabel("RSI Asset2")
# ax3.set_title("RSI Values (KDE heatmap)")

# # Normal RSI 3D surface
# ax4 = fig.add_subplot(2,2,4, projection='3d')
# ax4.plot_surface(X_rsi, Y_rsi, Z_rsi, cmap="plasma", edgecolor="none", alpha=0.9)
# ax4.set_xlabel("RSI Asset1")
# ax4.set_ylabel("RSI Asset2")
# ax4.set_zlabel("Density")
# ax4.set_title("RSI Values (KDE 3D surface)")

# # plt.tight_layout()
# # plt.show()



def region_density(kde, x_range, y_range, n=2000):
    """Sample KDE values in a rectangular region and return the mean density."""
    xs = np.random.uniform(x_range[0], x_range[1], n)
    ys = np.random.uniform(y_range[0], y_range[1], n)
    pts = np.vstack([xs, ys])
    return np.mean(kde(pts))

# Middle region around (0.5,0.5)
mid_score = region_density(kde_copula, (0.1, 0.9), (0.1, 0.9))

# Lower tail region near (0.05,0.05)
lower_score = region_density(kde_copula, (0.0, 0.1), (0.0, 0.1))

# Upper tail region near (0.95,0.95)
upper_score = region_density(kde_copula, (0.9, 1.0), (0.9, 1.0))

print("Copula KDE density (regional averages):")
print(f"  Middle region: {mid_score:.6f}")
print(f"  Lower tail: {lower_score:.6f}")
print(f"  Upper tail: {upper_score:.6f}")
print(f"  Ratio of tails to middle: {(lower_score + upper_score) / mid_score:.2f}")



# RSI difference
rsi_diff = (price_df['rsi1'] - price_df['rsi2']).fillna(0)
rsi_diff = rsi_diff.rolling(window=10).mean()

# Correlation matrix
rsi_corr = price_df[['rsi1', 'rsi2']].corr().iloc[0,1]

spread = np.log(price_df['Asset1']) - np.log(price_df['Asset2'])
# rolling_z = (spread.rolling(window=200).mean() - spread) / spread.rolling(window=200).std()
# rolling_z = rolling_z.rolling(window=10).mean()

# RSI1 vs RSI2
# --- Plotting ---
fig, ax = plt.subplots(4, 1, figsize=(14,12), sharex=True)

# RSI1 vs RSI2
ax[0].plot(price_df.index, price_df['rsi1'], label="RSI Asset1", color="blue", alpha=0.8)
ax[0].plot(price_df.index, price_df['rsi2'], label="RSI Asset2", color="orange", alpha=0.8)
ax[0].axhline(70, color="red", linestyle="--", alpha=0.5)
ax[0].axhline(30, color="green", linestyle="--", alpha=0.5)
ax[0].set_ylabel("RSI")
ax[0].set_title("RSI of Asset1 and Asset2")
ax[0].legend()

# RSI Difference
ax[1].plot(price_df.index, rsi_diff, label="RSI1 - RSI2", color="purple")
ax[1].axhline(0, color="white", linestyle="--", alpha=0.7)
ax[1].set_ylabel("RSI Diff")
ax[1].set_title("Difference in RSI (Asset1 - Asset2)")
ax[1].legend()

# Rolling correlation
rolling_corr = price_df['rsi1'].rolling(200).corr(price_df['rsi2'])
ax[2].plot(price_df.index, rolling_corr, label="Rolling Corr (200)", color="teal")
ax[2].axhline(rsi_corr, color="red", linestyle="--", label=f"Full Corr = {rsi_corr:.2f}")
ax[2].set_ylabel("Correlation")
ax[2].set_title("RSI Correlation")
ax[2].legend()

# Rolling Z-score of spread
ax[3].plot(price_df.index, spread, label="(Spread)", color="darkorange")
# ax[3].axhline(0, color="white", linestyle="--", alpha=0.6)
# ax[3].axhline( 2, color="red", linestyle="--", alpha=0.5)
# ax[3].axhline(-2, color="green", linestyle="--", alpha=0.5)
ax[3].set_ylabel("Spread")
ax[3].set_xlabel("Time")
ax[3].set_title("Spread")
ax[3].legend()

plt.tight_layout()
plt.show()

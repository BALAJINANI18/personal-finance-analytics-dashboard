# ================================
# BACKEND (MUST BE FIRST)
# ================================
import matplotlib
matplotlib.use("TkAgg")

# ================================
# IMPORTS
# ================================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("expenses.csv")
df["Date"] = pd.to_datetime(df["Date"])

# ================================
# KPI CALCULATIONS
# ================================
total_expense = df["Amount"].sum()
avg_expense = df["Amount"].mean()
category_sum = df.groupby("Category")["Amount"].sum()
top_category = category_sum.idxmax()

# ================================
# DAILY + ML
# ================================
daily = df.groupby("Date")["Amount"].sum().reset_index()
daily["Day"] = np.arange(len(daily))

X = daily[["Day"]]
y = daily["Amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
daily["Predicted"] = model.predict(X)
r2 = model.score(X_test, y_test)

# ================================
# GRADIENT BACKGROUND FUNCTION
# ================================
def add_gradient_background(fig):
    ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
    ax_bg.axis("off")

    gradient = np.linspace(0, 1, 256).reshape(256, 1)

    top = np.array([230, 240, 255]) / 255      # very light blue
    bottom = np.array([180, 205, 255]) / 255   # soft blue

    gradient_rgb = gradient * bottom + (1 - gradient) * top
    gradient_rgb = gradient_rgb.reshape(256, 1, 3)

    ax_bg.imshow(gradient_rgb, aspect="auto", extent=[0, 1, 0, 1])

# ================================
# FIGURE + SUBPLOTS
# ================================
fig, axes = plt.subplots(3, 4, figsize=(16, 10))

# 🔥 REMOVE WHITE BACKGROUND COMPLETELY
fig.patch.set_alpha(0)

# ADD GRADIENT
add_gradient_background(fig)

# MAKE ALL AXES TRANSPARENT
for row in axes:
    for ax in row:
        ax.set_facecolor("none")

# ================================
# TITLE
# ================================
fig.suptitle(
    "Personal Finance Analytics Dashboard",
    fontsize=20,
    fontweight="bold",
    color="#0F172A"
)

# ================================
# KPI CARD FUNCTION
# ================================
def kpi(ax, title, value, color):
    ax.axis("off")
    ax.text(
        0.5, 0.5,
        f"{title}\n{value}",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="#FFFFFF",
            edgecolor=color,
            linewidth=2
        )
    )

# ================================
# KPI ROW
# ================================
kpi(axes[0, 0], "Total Expense (₹)", total_expense, "#2563EB")
kpi(axes[0, 1], "Average Expense (₹)", f"{avg_expense:.2f}", "#16A34A")
kpi(axes[0, 2], "Top Category", top_category, "#F59E0B")
kpi(axes[0, 3], "ML R² Score", f"{r2:.2f}", "#DC2626")

# ================================
# BAR CHART
# ================================
ax = axes[1, 0]
ax.set_facecolor("#FFFFFF")
ax.bar(category_sum.index, category_sum.values, color="#2563EB")
ax.set_title("Category-wise Expense")
ax.set_ylabel("Amount (₹)")
ax.tick_params(axis="x", rotation=45)
ax.grid(True, linestyle="--", alpha=0.4)

for i, v in enumerate(category_sum.values):
    ax.text(i, v + 30, v, ha="center", fontweight="bold")

# ================================
# PIE CHART
# ================================
ax = axes[1, 1]
ax.set_facecolor("#FFFFFF")
ax.pie(
    category_sum.values,
    labels=category_sum.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=["#2563EB", "#16A34A", "#F59E0B", "#9333EA", "#DC2626"]
)
ax.set_title("Expense Distribution")
ax = axes[1, 2]
ax.set_facecolor("#FFFFFF")
ax.plot(daily["Date"], daily["Amount"], marker="o", color="#2563EB")
ax.set_title("Daily Expense Trend")
ax.set_ylabel("Amount (₹)")
ax.tick_params(axis="x", rotation=45)
ax.grid(True, linestyle="--", alpha=0.4)
ax = axes[1, 3]
ax.set_facecolor("#FFFFFF")
ax.plot(daily["Date"], daily["Amount"], label="Actual", marker="o", color="#2563EB")
ax.plot(daily["Date"], daily["Predicted"], label="Predicted", linestyle="--", color="#F59E0B")
ax.set_title("ML: Actual vs Predicted")
ax.legend()
ax.tick_params(axis="x", rotation=45)
ax.grid(True, linestyle="--", alpha=0.4)
for j in range(4):
    axes[2, j].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show(block=True)
input("Press Enter to close dashboard...")
plt.savefig(
    "finance_dashboard_linkedin.png",
    dpi=200,
    bbox_inches="tight",
    facecolor="#EAF2FF"  
)


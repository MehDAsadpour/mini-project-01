import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.style.use("dark_background")
plt.figure(figsize=(10, 6))

def load_data(path):
    return pd.read_csv(path)


def check_dataset_info(df):
    print("Dataset shape:")
    print(df.shape)

    print("\nDataset info:")
    print(df.info())


def check_missing_values(df):
    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nTotal Missing Values:")
    print(df.isnull().sum().sum())



def check_duplicates(df):
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")

    df = df.drop_duplicates()

    print(f"Shape after removing duplicates: {df.shape}")

    return df


def check_class_distribution(df):
    print("\nClass distribution:")
    print(df["Class"].value_counts())

    print("\nClass percentage:")
    print(df["Class"].value_counts(normalize=True) * 100)


def analyze_amount(df):
    print("\nAmount statistics:")
    print(df["Amount"].describe())

    print("\nZero Amount:")
    print((df["Amount"] == 0).sum())

    print("\nAmount by Class:")
    print(df.groupby("Class")["Amount"].describe())


def analyze_time(df):
    print("\nTime statistics:")
    print(df["Time"].describe())

    print("\nTime by Class:")
    print(df.groupby("Class")["Time"].describe())


def correlation_analysis(df):
    print("\nCorrelation with target:")
    print(df.corr(numeric_only=True)["Class"].sort_values())


def descriptive_statistics(df):
    print("\nDescriptive statistics:")
    print(df.describe().T)

def split_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def plot_class_distribution(df):
    df["Class"].value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()


def plot_amount_distribution(df):
    plt.hist(df["Amount"], bins=50)
    plt.title("Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.show()


def plot_time_distribution(df):
    plt.hist(df["Time"], bins=50)
    plt.title("Time Distribution")
    plt.xlabel("Seconds")
    plt.ylabel("Frequency")
    plt.show()


def plot_amount_by_class(df):
    df.boxplot(column="Amount", by="Class")
    plt.title("Amount by Class")
    plt.suptitle("")
    plt.show()


def plot_correlation_heatmap(df):
    plt.figure(figsize=(14, 10))

    corr = df.corr(numeric_only=True)

    plt.imshow(corr, cmap="coolwarm", aspect="auto")

    plt.colorbar()

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    df = load_data("data/creditcard.csv")

    check_dataset_info(df)

    check_missing_values(df)

    df = check_duplicates(df)

    check_class_distribution(df)

    analyze_amount(df)

    analyze_time(df)

    correlation_analysis(df)

    descriptive_statistics(df)

    plot_class_distribution(df)

    plot_amount_distribution(df)

    plot_time_distribution(df)

    plot_amount_by_class(df)

    plot_correlation_heatmap(df)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_scaled, X_test_scaled, scaler = scale_data(
    X_train,
    X_test)
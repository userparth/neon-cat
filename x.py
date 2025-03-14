
def myfunc():
    # Load CSV file
    data = pd.read_csv("fastreviews.csv")  # Ensure the correct path

    # Check data types
    print("Data Types:\n", data.dtypes)

    # Convert categorical columns using One-Hot Encoding
    categorical_columns = ["Agent Name", "Location", "Order Type",
                           "Customer Feedback Type", "Price Range",
                           "Discount Applied", "Product Availability"]

    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Encode 'Order Accuracy' column (Correct/Incorrect -> 1/0)
    if "Order Accuracy" in data.columns:
        data["Order Accuracy"] = data["Order Accuracy"].map({"Correct": 1, "Incorrect": 0})

    # Verify the changes
    print("Processed Data Sample:\n", data.head())

    # Show basic statistics
    print("Data Statistics:\n", data.describe())

    # Check missing values
    print("Missing Values:\n", data.isnull().sum())

    # Drop non-numeric columns before correlation
    numeric_data = data.select_dtypes(include=['number'])

    # Generate heatmap only for numerical features
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    return data  # Returning processed data for further steps
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def read_csv(abs_filename):
    df = pd.read_csv(filepath_or_buffer=abs_filename)
    #print(df.head())
    return df

def preprocessing(data_for_scaling):
    "Scaling the input features to bring all values within the same range between 0 to 1."
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_scaling)
    return scaled_data

def get_data(df):
    "Reading the specific features from the dataframe df received as user argument."
    X = df[['R&D Spend','Administration','Marketing Spend','State']]
    X = pd.get_dummies(X, drop_first=True)
    y = df['Profit'] 
    return X, y

def explore_dataset(df):
    print(f"Total columns in the csv file are: {df.shape}")
    
def generate_pairplot(df, hue):
    sns.pairplot(data=df, hue=hue)
    plt.savefig("PairplotOfInputData.png")

def split_training_test_data(X, y, test_size, random_state = 42):
    "Splitting the complete data into 2 sets for training and testing."
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def create_train_model(X, y, optimization_algo):
    """
        Defined multiple different training models.
        Based on the received user_input value of optimization_algo parameter, the corresponding model is created.
        In model.fit(...) statement is actually where the model is trained labeled input data, and its input parameters,cost and loss functions are calculated.
    """
    if optimization_algo == "Linear":
        optimization_algo_used = "LinearRegression"
        model = LinearRegression()
    elif optimization_algo == "Ridge":
        optimization_algo_used = "Ridge"
        model = Ridge()
    elif optimization_algo == "Lasso":
        optimization_algo_used = "Lasso"
        model = Lasso()
    elif optimization_algo == "ElasticNet":
        optimization_algo_used = "ElasticNet"
        model = ElasticNet()
    elif optimization_algo == "GradientBoost":
        optimization_algo_used = "GradientBoostingRegressor"
        model = GradientBoostingRegressor()
    else:
        optimization_algo = "Unknown algo"
        raise ValueError("Unknown optimization algo is received as input.")
    
    model.fit(X=X, y=y)
    return model, optimization_algo_used

def make_predictions(model,input_data):
    "Our trained model is actually making predictions here."
    return model.predict(input_data)

def plot_comparision(y_test, y_pred, used_optimization_algo):
    """
        1. Plotting comparision plots across actual vs predicted value of profit.
        
        2. We are checking if y_test has a .values attribue, which is true, if its a pandas series or dataframe.
           'y_test.values' converts it into a numpy array(which is needed for proper plotting and indexing consistency).
        3. It is done because, matplotlib works best with numpy arrays.
    """
    if hasattr(y_test, "values"):
        y_test = y_test.values   
    plt.figure(figsize=(10,6))
    plt.plot(range(len(y_test)), y_test,color='red', linewidth = 2, label='Actual')
    plt.plot(range(len(y_test)), y_pred, color='blue', linewidth=2, linestyle='dotted', label='Predicted')
    plt.title(f"Actual vs Predicted Profit with {used_optimization_algo}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{used_optimization_algo}ComparisionPlot.png')

def performance_measure(y_test, y_pred, used_optimization_algo, model_performance_results):
    """
        1. Calculating the Mean Squared Error and R2 score perfromance metrics across different model predictions.
        2. Storing the results for each algorithm into model_performance_results dictionary.
        3. This dictionary data will be later used for generating plots.
    """
    MSE = mean_squared_error(y_true=y_test, y_pred=y_pred)
    R2_score = r2_score(y_true=y_test, y_pred=y_pred)
    model_performance_results['Algorithm'].append(used_optimization_algo)
    model_performance_results['MSE'].append(MSE)
    model_performance_results['R2_score'].append(R2_score)
    print(f"The mean squared error with {used_optimization_algo} is: {MSE}")
    print(f"The r2 score with {used_optimization_algo} is: {R2_score}\n\n")

def plot_model_performance(result_data):
    "Generating plots for MSE and R2 across metrics for different models."

    df = pd.DataFrame(result_data)
    # Plot MSE
    plt.figure(figsize=(10,5))
    ax1 = sns.barplot(y='MSE', data=df,hue='Algorithm')
    ax1.legend(title='Algorithm', loc='upper right', bbox_to_anchor=(0.99,0.98), fontsize='small', title_fontsize='medium')
    plt.title("Mean Squared Error by Algorithms")
    plt.xlabel("Algorithm")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig("All_Models_Comparative_MSE_Plot.png")

    # Plot R2 score
    plt.figure(figsize=(10,5))
    ax2 = sns.barplot(y = 'R2_score', data=df, hue='Algorithm')
    ax2.legend(title='Algorithm', loc='upper right', bbox_to_anchor=(1.18, 1.20), fontsize='small', title_fontsize='medium')
    plt.xlabel("Algorithm")
    plt.ylabel("R2 score")
    plt.title("R2 score by Algorithms")
    plt.tight_layout()
    plt.savefig("All_Models_Comparative_R2_Score_Plot.png")

def main():
    df = read_csv("archive\\50_Startups.csv")
    explore_dataset(df=df)
    X, y = get_data(df=df)
    X = preprocessing(data_for_scaling=X)
    X_train, X_test, y_train, y_test = split_training_test_data(X=X, y=y, test_size=0.2, random_state=42)
    model_performance_results = {"Algorithm": [], "MSE": [], "R2_score":[]}
    for optimization_algo in ["Linear", "Ridge","Lasso", "ElasticNet", "GradientBoost"]:
        model, used_optimization_algo = create_train_model(X=X_train, y=y_train, optimization_algo=optimization_algo)
        y_pred = make_predictions(model=model, input_data=X_test)
        plot_comparision(y_test=y_test, y_pred=y_pred, used_optimization_algo=used_optimization_algo)
        performance_measure(y_test=y_test, y_pred=y_pred, used_optimization_algo=used_optimization_algo, model_performance_results = model_performance_results)
    plot_model_performance(result_data= model_performance_results)


if __name__ == "__main__":
    main()
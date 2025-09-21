import main_func as mf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# dataset = mf.load_data('final_data_Reduce E,C,F ,SD.csv')
# X_train,y_train,X_test,y_test = mf.data_split(dataset)
# dataset = mf.Standardize_data(dataset)

def PCA_Data(x_train,x_test):
    # Make an instance of the Model
    pca = PCA(.95)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    # pca.n_components_
    return x_train, x_test


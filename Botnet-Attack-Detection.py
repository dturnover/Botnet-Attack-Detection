import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix
import streamlit as st
import os


@st.cache()
def import_dataset(directory):
    # hold data for each device in separate lists
    devices = [[], [], [], [], [], [], [], [], []]

    # iterate through each directory, subdirectory and file
    itr = -3
    for root, subdirectories, files in os.walk(directory):
        for file in files:
            # check if a new subdirectory was entered
            if len(subdirectories) != 0:
                itr += 1

            # check if the file is a .csv
            filename = os.path.join(root, file)
            if filename.endswith(".csv"):

                # read the .csv into a dataframe
                temp_df = pd.read_csv(filename)

                # keep benign devices (training data) at the front of the list
                if "benign" in filename:
                    devices[itr].insert(0, temp_df)
                    print(filename, 'inserted')
                else:
                    devices[itr].append(temp_df)
                    print(filename, 'appended')

    return devices


def ensure_stationarity(devices):
    stationary_features = [[], [], [], [], [], [], [], [], []]

    for i in range(len(devices)):
        for j in range(len(devices[i])):
            temp_df = devices[i][j].copy(deep=True)

            for col in temp_df.columns:
                # difference every column of every dataframe to make data stationary
                temp_df[col] = temp_df[col].diff()

            # drop rows with na values
            temp_df.dropna(inplace=True)
            stationary_features[i].append(temp_df)

    return stationary_features


def scale_features(stationary_features):
    scaled_features = [[], [], [], [], [], [], [], [], []]

    for i in range(len(stationary_features)):
        for j in range(len(stationary_features[i])):

            # standarize data so it is all on a similar scale
            standardize = stationary_features[i][0].values
            standardize = StandardScaler().fit_transform(standardize)

            if j == 0:
                scaled_features[i].insert(0, standardize)
            else:
                scaled_features[i].append(standardize)

            # delete stationary data to free up memory
            del stationary_features[i][0]

    return scaled_features


@st.cache()
def principal_component_analysis(scaled_features, num_components):
    components = [[], [], [], [], [], [], [], [], []]

    pca = PCA(n_components=num_components)

    for i in range(len(scaled_features)):
        for j in range(len(scaled_features[i])):

            # perform principal component analysis to improve speed and generalization
            principalComponents = pca.fit_transform(scaled_features[i][0])
            principal_df = pd.DataFrame(data=principalComponents)

            if j == 0:
                components[i].insert(0, principal_df)
            else:
                components[i].append(principal_df)

            # clear up memory
            del scaled_features[i][0]

    return components, pca.explained_variance_ratio_


def train_opt_test_split(component):
    # split the training data into three equal size subsets
    size = len(component.index)
    split = round(size * 0.33)

    # training set trains each model
    train_data = component.iloc[:split, :]
    # the optimize set tests the model during optimization
    optimize_data = component.iloc[split:split * 2, :]
    # the test set tests the model after optimization
    test_data = component.iloc[split * 2:, :]

    return train_data, optimize_data, test_data


@st.cache()
def grid_search_optimize(train_data, opt_data):
    # define parameters worth adjusting during optimization
    n_neighbors_dict = [10, 15, 20, 25, 30]
    p_dict = [1, 2, 3, 4]

    lowest_score = 1000
    best_parameters = []

    for d1 in n_neighbors_dict:
        for d2 in p_dict:
            # initialize the local outlier factor model
            opt_model = LocalOutlierFactor(n_neighbors=d1,
                                           p=d2,
                                           contamination=0.001,
                                           n_jobs=-1,
                                           novelty=True)

            # fit the model to the training data
            opt_model.fit(train_data)

            # perform anomaly detection on optimization data
            opt_predictions = opt_model.predict(opt_data)
            opt_pred_df = pd.DataFrame(opt_predictions)

            # count the number of false positives (all predictions should be 1 not -1)
            false_positives = len(opt_pred_df[opt_pred_df[0] == -1])

            # the model with the lowest number of false positives is said to perform best
            if false_positives < lowest_score:
                best_parameters = [d1, d2]
                lowest_score = false_positives

    return best_parameters


def train_models(data, inputs):
    models = []
    predictions = []

    for i in range(len(data)):
        # divide the training data into subsets
        train_data, optimize_data, test_data = train_opt_test_split(data[i][0])

        # use grid search optimization to find good parameters for each model unless the user specified their own parameters
        if len(inputs) == 0:
            parameters = grid_search_optimize(train_data, optimize_data)
            print("Selected parameters:", parameters[0], parameters[1])
        else:
            parameters = inputs

        # initialize local outlier factor with chosen parameters
        LOF_model = LocalOutlierFactor(n_neighbors=parameters[0],
                                       p=parameters[1],
                                       contamination=0.00001,
                                       n_jobs=-1,
                                       novelty=True)

        # train model and append it to list of all models (one for each device)
        LOF_model.fit(train_data)
        models.append(LOF_model)

        # see how well it performs
        prediction = LOF_model.predict(test_data)
        prediction_df = pd.DataFrame(prediction)
        predictions.append(prediction_df)

        print("False positives:", len(prediction_df[prediction_df[0] == -1]), '\n')

    return models, predictions


def test_infected_devices(models, data):
    attack_predictions = [[], [], [], [], [], [], [], [], []]

    for i in range(len(models)):
        for j in range(1, len(data[i])):
            # use the model of each device to predict whether corresponding devices are infected
            attack_preds = models[i].predict(data[i][j])
            attack_preds_df = pd.DataFrame(attack_preds)

            attack_predictions[i].append(attack_preds_df)

    return attack_predictions


# define containers for web app
header = st.container()
dataset = st.container()
preprocessing = st.container()
modelTraining = st.container()
conlusion = st.container()

with header:
    st.title("AAI 530 Final Project")
    st.header("IoT Botnet Attack Detection")
    st.markdown("The dataset I chose recorded network traffic across nine IoT devices. IoT devices are more prone to hacking than personal computers. Many are deployed without changing factory settings which leaves their credentials open to brute-force password cracking. Once hacked, the IoT device becomes a part of a botnet, and can be used for malicious activity. The extra networking/computing demands from the botnet are difficult for the already tightly constrained IoT device. Performance will take a hit as a result. To solve this problem I developed a semi-supervised classification method for determining whether IoT devices are infected using anomaly detection")

with dataset:
    st.header("Dataset Details")
    st.markdown("Packet information for nine IoT devices was captured using port mirroring. Nine .csv files capture what uncompromised for each device. These files should be classified as 0 (false). The remaining 80 .csv files capture various attack types (tcp/udp flooding, bruteforce, junk, etc), each should be classified as 1 (true). Features were engineered using various statistics computed on packet streams. Weight, mean, standard deviation, radius, magnitude, and covariance were computed on stream aggregations which described:")
    st.markdown("H: Recent traffic from the packet's host")
    st.markdown("HH: Recent traffic going from the packet's host to the destination's host")
    st.markdown("HpHp: Recent traffic going from the packet's host and port, to the packet's destination's host and port")
    st.markdown("HH_jit: Jitter of traffic going from the packet's host to the destination's host")
    st.markdown("In all, 115 features were extracted for each device.")
    st.write("**Note** all plots and metrics displayed represent a single device")

    # embed program inside containers
    path = r"C:\Users\muggs\Desktop\AAI-530 Data Analytics and IoT\Assignments\Final Project"
    device_data = import_dataset(path)

    # display dataset format
    st.write("**Summary**", device_data[0][0].head())

    # plot one feature for an infected, and corresponding benign device
    selected_feature = st.text_input("Feature to display")

    # create columns for the benign signal plot and infected signal plot
    lcol_1, rcol_1 = st.columns(2)

    # ensure valid input
    if selected_feature in device_data[0][0].columns:
        # plot benign feature signal
        lcol_1.write("**Benign device**")
        fig_1 = plt.figure()
        device_data[0][0][selected_feature].plot()
        lcol_1.pyplot(fig_1)

        # plot inected feature signal
        rcol_1.write("**Infected device**")
        fig_2 = plt.figure()
        device_data[0][1][selected_feature].plot()
        rcol_1.pyplot(fig_2)

    elif selected_feature != "":
        st.markdown("Invalid column name")


with preprocessing:
    st.header("Preprocessing Steps")
    st.markdown("Principal component analysis is deployed to reduce the number of features. Speed and generalizability drastically increase as a result. In order to conduct PCA effectively the dataset is differenced to ensure stationarity, and standardized to stabilize variance. This helps PCA clearly distinguish its components.")

    # make data trend stationary
    stationary_data = ensure_stationarity(device_data)

    # scale features so that data is standardized
    standardized_data = scale_features(stationary_data)

    lcol_2, rcol_2 = st.columns(2)

    # plot the same feature after data preprocessing is finished
    if selected_feature in device_data[0][0].columns:
        fig_3 = plt.figure()
        display_transf = pd.DataFrame(standardized_data[0][0], columns=device_data[0][0].columns)
        display_transf[selected_feature].plot()
        lcol_2.write("**Benign signal after transformations**")
        lcol_2.pyplot(fig_3)

        fig_4 = plt.figure()
        display_transf = pd.DataFrame(standardized_data[0][1], columns=device_data[0][1].columns)
        display_transf[selected_feature].plot()
        rcol_2.write("**Infected signal after transformations**")
        rcol_2.pyplot(fig_4)

    # use principal component analysis as a means to cluster data
    n_comps = lcol_2.slider("Number of PCA components", min_value=1, max_value=20, value=5)
    component_data, explained_variance = principal_component_analysis(standardized_data, n_comps)

    rounded_ev = [round(num, 3) for num in explained_variance]
    with rcol_2:
        st.write("Explained variation per principal component: ", rounded_ev)
        st.metric("Total variance:", round(sum(explained_variance), 3))


with modelTraining:
    st.header("Model Training")
    st.markdown("One local outlier factor model is trained on the benign data of each device to gain an understanding of the device's baseline behavior. The benign data is divided into three equal size subsets: a training subset, an optimization subset, and a testing subset. The training subset trains each model, the optimization subset evaluates its performance during parameter optimization, and the testing subset evaluats the overall performance on benign data post-optimization. Afterwards the model is tested on the infected data. The local outlier factor performs anomaly detection to pick out suspicious activity that bears little resemblance to the device's typical behavior. If any anomolies are detected the dataset is predicted to be infected.")

    lcol_3, rcol_3 = st.columns(2)

    # optimize models with user input or grid-search optimization
    with lcol_3:
        st.markdown("**Optimization**")
        num_n = lcol_3.slider("Number of neighbors in LOF", min_value=1, max_value=40)
        mink = lcol_3.slider("Minkowski metric (for computing distances)", min_value=1, max_value=10)

        params = [num_n, mink]

        auto = lcol_3.checkbox("Auto", value=True)
        if auto:
            # train one model for every device using grid-search optimization
            LOF_models, benign_preds = train_models(component_data, [])
        else:
            # train one model for every device using input parameters
            LOF_models, benign_preds = train_models(component_data, params)

    with rcol_3:
        # run predictions for malicious data
        preds = test_infected_devices(LOF_models, component_data)

        # display the number of predicted attacks in each infected device
        for k in range(len(preds)):
            for m in range(len(preds[k])):
                print("Number of attacks detected:", len(preds[k][m][preds[k][m][0] == -1]), '/',
                      len(preds[k][m]), ':',
                      round(len(preds[k][m][preds[k][m][0] == -1]) / len(preds[k][m]), 4), '%')

        # 0 attacks detected means a negative prediction
        # 1 or more detected attacks means a positive prediction
        y_hat = []
        y = []

        # count the number of true positives, false negatives, etc
        for k in range(len(benign_preds)):
            if len(benign_preds[k][benign_preds[k][0] == -1]) > 0:
                y_hat.append(1)
            else:
                y_hat.append(0)

            y.append(0)

        for k in range(len(preds)):
            for m in range(len(preds[k])):
                if len(preds[k][m][preds[k][m][0] == -1]) > 0:
                    y_hat.append(1)
                else:
                    y_hat.append(0)

                y.append(1)

        # display which devices were correctly and incorrectly classified (as infected (postive) or benign (negative))
        cm = confusion_matrix(y, y_hat)

        st.write("**Confusion Matrix**", cm)

        # calculate performance metrics
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        F1_score = 2 * (sensitivity * precision) / (sensitivity + precision)

        st.metric("Accuracy:", round(accuracy, 3))
        st.metric("Precision:", round(precision, 3))
        st.metric("Sensitivity:", round(sensitivity, 3))
        st.metric("Specificity:", round(specificity, 3))
        st.metric("F1 Score:", round(F1_score, 3))


with conlusion:
    st.header("Conclusion")
    st.write("The model performs exceptionally well at classifying devices when using the proper settings.")
    st.write("**To obtain the best results:**")
    st.write("Set the number of PCA components equal to five")
    st.write("Check the box marked 'auto' to use grid-search optimize on each model")
    st.write("The performance metrics on the test data with these settings is:")
    st.write("Accuracy: 98.9%")
    st.write("Precision: 98.8%")
    st.write("Sensitivity: 100%")
    st.write("Specificity: 88.9%")
    st.write("F1 score: 99.4%")

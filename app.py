import streamlit as st
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime
from sklearn.calibration import calibration_curve
#import xmltodict
#from mitosheet.streamlit.v1 import spreadsheet
from pandas import json_normalize
#from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_lottie import st_lottie
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Set plot style
sns.set(color_codes=True)

# Page d'accueil
st.set_page_config(
    page_title="costumer churn prediction",
    page_icon="🌉",
    layout="wide"
)

# Définissez la couleur de fond
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_book = load_lottieurl("https://assets4.lottiefiles.com/temp/lf20_aKAfIn.json")
st_lottie(lottie_book, speed=1, height=200, key="initial")

# Ajouter un titre
st.title("Projet de Machine Learning pour Prédire le Churn chez PowerCo : Vers une Rétention Client Plus Efficace")
st.markdown(""" Ce projet s'inscrit dans la volonté de PowerCo d'optimiser sa stratégie de rétention des clients PME 
et de renforcer sa position sur le marché de l'énergie.
 
Je vous invite à me suivre dans ma démarche de résolution de cette problématique.""")
st.markdown("<p style='text-align: right;'> Réalisé par Blandine JATSA NGUETSE </p>", unsafe_allow_html=True)


# definition des fonctions utiles
def annotate_stacked_bars(ax, pad=0.99, colour="white", textsize=8):
    """
    Add value annotations to the bars
    """
    annotations = []

    # Iterate over the plotted rectangles/bars
    for p in ax.patches:
        # Calculate annotation
        value = str(round(p.get_height(), 1))
        # If value is 0 do not annotate
        if value == '0.0':
            continue
        annotation = ax.annotate(
            value,
            ((p.get_x() + p.get_width() / 2) * pad - 0.05, (p.get_y() + p.get_height() / 2) * pad),
            color=colour,
            size=textsize
        )
        annotations.append(annotation)

    return annotations

def plot_stacked_bars(dataframe, title_, size_=(18, 10), rot_=0, legend_="upper right"):
    """
    Plot stacked bars with annotations
    """
    fig, ax = plt.subplots(figsize=size_)

    dataframe.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        rot=rot_,
        title=title_
    )

    # Annotate bars
    annotations = annotate_stacked_bars(ax, textsize=18)
    # Rename legend
    ax.legend(["Retention", "Churn"], loc=legend_)
    # Labels
    ax.set_ylabel("Base de clients (%)")

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

def display_data_types_info(dataframe):
    # Obtenir les types de données pour chaque variable
    data_types_info = dataframe.dtypes.reset_index()
    data_types_info.columns = ['Variable', 'Type']

    # Afficher les informations dans Streamlit
    #st.write("#### Informations sur les types de données:")
    st.table(data_types_info)

def plot_distribution(dataframe, column, bins_=50):
    """
    Plot variable distirbution in a stacked histogram of churned or retained company
    """
    # Create a temporal dataframe with the data to be plot
    temp = pd.DataFrame({"Retention": dataframe[dataframe["churn"]==0][column],
                         "Churn":dataframe[dataframe["churn"]==1][column]})

    # Plot the histogram
    fig, ax = plt.subplots(figsize=(8, 10))
    temp[["Retention","Churn"]].plot(kind='hist', bins=bins_, ax=ax, stacked=True)
    # X-axis label
    ax.set_xlabel(column)
    # Change the x-axis to plain style
    ax.ticklabel_format(style='plain', axis='x')
    st.pyplot(fig)

# importation des données
client_df = pd.read_csv('./client_data.csv')
price_df = pd.read_csv('./price_data.csv')

#churn
churn = client_df[['id', 'churn']]
churn.columns = ['Companies', 'churn']
churn_total = churn.groupby(churn['churn']).count()
churn_percentage = churn_total / churn_total.sum() * 100

#Sales channel
channel = client_df[['id', 'channel_sales', 'churn']]
channel = channel.groupby([channel['channel_sales'], channel['churn']])['id'].count().unstack(level=1).fillna(0)
channel_churn = (channel.div(channel.sum(axis=1), axis=0) * 100).sort_values(by=[1], ascending=False)


forecast = client_df[
    ["id", "forecast_cons_12m",
    "forecast_cons_year","forecast_discount_energy","forecast_meter_rent_12m",
    "forecast_price_energy_off_peak","forecast_price_energy_peak",
    "forecast_price_pow_off_peak","churn"
    ]
]
contract_type = client_df[['id', 'has_gas', 'churn']]
margin = client_df[['id', 'margin_gross_pow_ele', 'margin_net_pow_ele', 'net_margin']]
power = client_df[['id', 'pow_max', 'churn']]

client_df["date_activ"] = pd.to_datetime(client_df["date_activ"], format='%Y-%m-%d')
client_df["date_end"] = pd.to_datetime(client_df["date_end"], format='%Y-%m-%d')
client_df["date_modif_prod"] = pd.to_datetime(client_df["date_modif_prod"], format='%Y-%m-%d')
client_df["date_renewal"] = pd.to_datetime(client_df["date_renewal"], format='%Y-%m-%d')
price_df['price_date'] = pd.to_datetime(price_df['price_date'], format='%Y-%m-%d')


mean_year = price_df.groupby(['id']).mean().reset_index()
mean_6m = price_df[price_df['price_date'] > '2015-06-01'].groupby(['id']).mean().reset_index()
mean_3m = price_df[price_df['price_date'] > '2015-10-01'].groupby(['id']).mean().reset_index()


# Combinez en un seul dataframe
mean_year = mean_year.rename(
    index=str,
columns={
                    "price_off_peak_var": "mean_year_price_off_peak_var",
                    "price_peak_var": "mean_year_price_peak_var",
                    "price_mid_peak_var": "mean_year_price_mid_peak_var",
                    "price_off_peak_fix": "mean_year_price_off_peak_fix",
                    "price_peak_fix": "mean_year_price_peak_fix",
                    "price_mid_peak_fix": "mean_year_price_mid_peak_fix"
                }
            )

mean_year["mean_year_price_off_peak"] = mean_year["mean_year_price_off_peak_var"] + mean_year["mean_year_price_off_peak_fix"]
mean_year["mean_year_price_peak"] = mean_year["mean_year_price_peak_var"] + mean_year["mean_year_price_peak_fix"]
mean_year["mean_year_price_mid_peak"] = mean_year["mean_year_price_mid_peak_var"] + mean_year["mean_year_price_mid_peak_fix"]


mean_6m = mean_6m.rename(
                index=str,
                columns={
                    "price_off_peak_var": "mean_6m_price_off_peak_var",
                    "price_peak_var": "mean_6m_price_peak_var",
                    "price_mid_peak_var": "mean_6m_price_mid_peak_var",
                    "price_off_peak_fix": "mean_6m_price_off_peak_fix",
                    "price_peak_fix": "mean_6m_price_peak_fix",
                    "price_mid_peak_fix": "mean_6m_price_mid_peak_fix"
                }
            )

mean_6m["mean_6m_price_off_peak"] = mean_6m["mean_6m_price_off_peak_var"] + mean_6m["mean_6m_price_off_peak_fix"]
mean_6m["mean_6m_price_peak"] = mean_6m["mean_6m_price_peak_var"] + mean_6m["mean_6m_price_peak_fix"]
mean_6m["mean_6m_price_mid_peak"] = mean_6m["mean_6m_price_mid_peak_var"] + mean_6m["mean_6m_price_mid_peak_fix"]

mean_3m = mean_3m.rename(
    index=str,
    columns={
                    "price_off_peak_var": "mean_3m_price_off_peak_var",
                    "price_peak_var": "mean_3m_price_peak_var",
                    "price_mid_peak_var": "mean_3m_price_mid_peak_var",
                    "price_off_peak_fix": "mean_3m_price_off_peak_fix",
                    "price_peak_fix": "mean_3m_price_peak_fix",
                    "price_mid_peak_fix": "mean_3m_price_mid_peak_fix"
                }
            )

mean_3m["mean_3m_price_off_peak"] = mean_3m["mean_3m_price_off_peak_var"] + mean_3m["mean_3m_price_off_peak_fix"]
mean_3m["mean_3m_price_peak"] = mean_3m["mean_3m_price_peak_var"] + mean_3m["mean_3m_price_peak_fix"]
mean_3m["mean_3m_price_mid_peak"] = mean_3m["mean_3m_price_mid_peak_var"] + mean_3m["mean_3m_price_mid_peak_fix"]

# Merge into 1 dataframe
price_features = pd.merge(mean_year, mean_6m, on='id')
price_features = pd.merge(price_features, mean_3m, on='id')
price_analysis = pd.merge(price_features, client_df[['id', 'churn']], on='id')
# display_data_types_info(price_features)
# display_data_types_info(price_analysis)
# Calcul de la matrice de corrélation
price_analysis_copy = price_analysis.drop('id', axis=1)
merged_data = pd.merge(client_df.drop(columns=['churn']), price_analysis, on='id')
#merged_data.to_csv('clean_data_after_eda.csv')

#Feature Engineering
df = merged_data.copy()
df["date_activ"] = pd.to_datetime(df["date_activ"], format='%Y-%m-%d')
df["date_end"] = pd.to_datetime(df["date_end"], format='%Y-%m-%d')
df["date_modif_prod"] = pd.to_datetime(df["date_modif_prod"], format='%Y-%m-%d')
df["date_renewal"] = pd.to_datetime(df["date_renewal"], format='%Y-%m-%d')
price_df = pd.read_csv('price_data.csv')
price_df["price_date"] = pd.to_datetime(price_df["price_date"], format='%Y-%m-%d')
# Group off-peak prices by companies and month
monthly_price_by_id = price_df.groupby(['id', 'price_date']).agg(
    {'price_off_peak_var': 'mean', 'price_off_peak_fix': 'mean'}).reset_index()

# Get january and december prices
jan_prices = monthly_price_by_id.groupby('id').first().reset_index()
dec_prices = monthly_price_by_id.groupby('id').last().reset_index()

# Calculate the difference
diff = pd.merge(dec_prices.rename(columns={'price_off_peak_var': 'dec_1', 'price_off_peak_fix': 'dec_2'}),
                    jan_prices.drop(columns='price_date'), on='id')
diff['offpeak_diff_dec_january_energy'] = diff['dec_1'] - diff['price_off_peak_var']
diff['offpeak_diff_dec_january_power'] = diff['dec_2'] - diff['price_off_peak_fix']
diff = diff[['id', 'offpeak_diff_dec_january_energy', 'offpeak_diff_dec_january_power']]
df = pd.merge(df, diff, on='id')

# Aggregate average prices per period by company
mean_prices = price_df.groupby(['id']).agg({
        'price_off_peak_var': 'mean',
        'price_peak_var': 'mean',
        'price_mid_peak_var': 'mean',
        'price_off_peak_fix': 'mean',
        'price_peak_fix': 'mean',
        'price_mid_peak_fix': 'mean'
    }).reset_index()

# Calculate the mean difference between consecutive periods
mean_prices['off_peak_peak_var_mean_diff'] = mean_prices['price_off_peak_var'] - mean_prices['price_peak_var']
mean_prices['peak_mid_peak_var_mean_diff'] = mean_prices['price_peak_var'] - mean_prices['price_mid_peak_var']
mean_prices['off_peak_mid_peak_var_mean_diff'] = mean_prices['price_off_peak_var'] - mean_prices[
        'price_mid_peak_var']
mean_prices['off_peak_peak_fix_mean_diff'] = mean_prices['price_off_peak_fix'] - mean_prices['price_peak_fix']
mean_prices['peak_mid_peak_fix_mean_diff'] = mean_prices['price_peak_fix'] - mean_prices['price_mid_peak_fix']
mean_prices['off_peak_mid_peak_fix_mean_diff'] = mean_prices['price_off_peak_fix'] - mean_prices[
        'price_mid_peak_fix']
columns = [
        'id',
        'off_peak_peak_var_mean_diff',
        'peak_mid_peak_var_mean_diff',
        'off_peak_mid_peak_var_mean_diff',
        'off_peak_peak_fix_mean_diff',
        'peak_mid_peak_fix_mean_diff',
        'off_peak_mid_peak_fix_mean_diff'
    ]
df = pd.merge(df, mean_prices[columns], on='id')

# Aggregate average prices per period by company
mean_prices_by_month = price_df.groupby(['id', 'price_date']).agg({
        'price_off_peak_var': 'mean',
        'price_peak_var': 'mean',
        'price_mid_peak_var': 'mean',
        'price_off_peak_fix': 'mean',
        'price_peak_fix': 'mean',
        'price_mid_peak_fix': 'mean'
    }).reset_index()

# Calculate the mean difference between consecutive periods
mean_prices_by_month['off_peak_peak_var_mean_diff'] = mean_prices_by_month['price_off_peak_var'] - \
                                                          mean_prices_by_month['price_peak_var']
mean_prices_by_month['peak_mid_peak_var_mean_diff'] = mean_prices_by_month['price_peak_var'] - mean_prices_by_month[
        'price_mid_peak_var']
mean_prices_by_month['off_peak_mid_peak_var_mean_diff'] = mean_prices_by_month['price_off_peak_var'] - \
                                                              mean_prices_by_month['price_mid_peak_var']
mean_prices_by_month['off_peak_peak_fix_mean_diff'] = mean_prices_by_month['price_off_peak_fix'] - \
                                                          mean_prices_by_month['price_peak_fix']
mean_prices_by_month['peak_mid_peak_fix_mean_diff'] = mean_prices_by_month['price_peak_fix'] - mean_prices_by_month[
        'price_mid_peak_fix']
mean_prices_by_month['off_peak_mid_peak_fix_mean_diff'] = mean_prices_by_month['price_off_peak_fix'] - \
                                                              mean_prices_by_month['price_mid_peak_fix']
# Calculate the maximum monthly difference across time periods
max_diff_across_periods_months = mean_prices_by_month.groupby(['id']).agg({
        'off_peak_peak_var_mean_diff': 'max',
        'peak_mid_peak_var_mean_diff': 'max',
        'off_peak_mid_peak_var_mean_diff': 'max',
        'off_peak_peak_fix_mean_diff': 'max',
        'peak_mid_peak_fix_mean_diff': 'max',
        'off_peak_mid_peak_fix_mean_diff': 'max'
    }).reset_index().rename(
    columns={
            'off_peak_peak_var_mean_diff': 'off_peak_peak_var_max_monthly_diff',
            'peak_mid_peak_var_mean_diff': 'peak_mid_peak_var_max_monthly_diff',
            'off_peak_mid_peak_var_mean_diff': 'off_peak_mid_peak_var_max_monthly_diff',
            'off_peak_peak_fix_mean_diff': 'off_peak_peak_fix_max_monthly_diff',
            'peak_mid_peak_fix_mean_diff': 'peak_mid_peak_fix_max_monthly_diff',
            'off_peak_mid_peak_fix_mean_diff': 'off_peak_mid_peak_fix_max_monthly_diff'
        }
    )
columns = [
        'id',
        'off_peak_peak_var_max_monthly_diff',
        'peak_mid_peak_var_max_monthly_diff',
        'off_peak_mid_peak_var_max_monthly_diff',
        'off_peak_peak_fix_max_monthly_diff',
        'peak_mid_peak_fix_max_monthly_diff',
        'off_peak_mid_peak_fix_max_monthly_diff'
    ]

df = pd.merge(df, max_diff_across_periods_months[columns], on='id')
#df['tenure'] = ((df['date_end'] - df['date_activ']) / np.timedelta64(1, 'Y')).astype(int)
df['tenure'] = ((df['date_end'] - df['date_activ']).dt.days / 365).astype(int)

def convert_months(reference_date, df, column):
    """
    Input a column with timedeltas and return months
    """
    time_delta = reference_date - df[column]
    months = (time_delta.dt.days / (365.25 / 12)).astype(int)
    return months

# Create reference date
reference_date = datetime(2016, 1, 1)

# Create columns
df['months_activ'] = convert_months(reference_date, df, 'date_activ')
df['months_to_end'] = -convert_months(reference_date, df, 'date_end')
df['months_modif_prod'] = convert_months(reference_date, df, 'date_modif_prod')
df['months_renewal'] = convert_months(reference_date, df, 'date_renewal')

df['has_gas'] = df['has_gas'].replace(['t', 'f'], [1, 0])
# Transform into categorical type
df['channel_sales'] = df['channel_sales'].astype('category')
df = pd.get_dummies(df, columns=['channel_sales'], prefix='channel')
df = df.drop(columns=['channel_sddiedcslfslkckwlfkdpoeeailfpeds', 'channel_epumfxlbckeskwekxbiuasklxalciiuu',
                          'channel_fixdbufsefwooaasfcxdxadsiekoceaa'])

# Transform into categorical type
df['origin_up'] = df['origin_up'].astype('category')
df = pd.get_dummies(df, columns=['origin_up'], prefix='origin_up')
df = df.drop(columns=['origin_up_MISSING', 'origin_up_usapbepcfoloekilkwsdiboslwaxobdp',
                          'origin_up_ewxeelcelemmiwuafmddpobolfuxioce'])
skewed = [
        'cons_12m',
        'cons_gas_12m',
        'cons_last_month',
        'forecast_cons_12m',
        'forecast_cons_year',
        'forecast_discount_energy',
        'forecast_meter_rent_12m',
        'forecast_price_energy_off_peak',
        'forecast_price_energy_peak',
        'forecast_price_pow_off_peak'
    ]
skewed_before = df[skewed].describe()

df["cons_12m"] = np.log10(df["cons_12m"] + 1)
df["cons_gas_12m"] = np.log10(df["cons_gas_12m"] + 1)
df["cons_last_month"] = np.log10(df["cons_last_month"] + 1)
df["forecast_cons_12m"] = np.log10(df["forecast_cons_12m"] + 1)
df["forecast_cons_year"] = np.log10(df["forecast_cons_year"] + 1)
df["forecast_meter_rent_12m"] = np.log10(df["forecast_meter_rent_12m"] + 1)
df["imp_cons"] = np.log10(df["imp_cons"] + 1)

skewed_after = df[skewed].describe()



df = df.drop(columns=['num_years_antig', 'forecast_cons_year'])
#df.to_csv("data_after_featuring.csv", index=False)

# Modélisation
df_model = df.copy()

#df_copy = df_model.drop(['Unnamed: 0', 'id', 'price_date_x', 'price_date_y', 'price_date', 'date_activ',
#                         'date_end', 'date_modif_prod', 'date_renewal'], axis=1)
df_copy = df_model.drop(['id', 'price_date_x', 'price_date_y', 'price_date', 'date_activ',
                         'date_end', 'date_modif_prod', 'date_renewal'], axis=1)
train_df = df_copy.copy()

# Separate target variable from independent variables
y = df_model['churn']
#X = df_model.drop(columns=['id', 'churn', 'Unnamed: 0', 'price_date_x', 'price_date_y', 'price_date', 'date_activ',
#                          'date_end', 'date_modif_prod', 'date_renewal' ])
X = df_model.drop(columns=['id', 'churn', 'price_date_x', 'price_date_y', 'price_date', 'date_activ',
                           'date_end', 'date_modif_prod', 'date_renewal' ])
#print(X.shape)
#print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestClassifier(
        n_estimators=1000
    )
model.fit(X_train, y_train)
predictions = model.predict(X_test)
tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
#st.dataframe(y_test.value_counts())
feature_importances = pd.DataFrame({
        'features': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=True).reset_index()

proba_predictions = model.predict_proba(X_test)
probabilities = proba_predictions[:, 1]
X_test = X_test.reset_index()
X_test.drop(columns='index', inplace=True)
X_test['churn'] = predictions.tolist()
X_test['churn_probability'] = probabilities.tolist()

#X_test.to_csv('./out_of_sample_data_with_predictions.csv')

# impact business de la remise

test_df = X_test.copy()
# Le chiffre d'affaires d'électricité pour chaque client est composé de la consommation d'énergie (quantité * prix)
# et du loyer du compteur
# (Le prix de l'électricité peut également jouer un rôle, mais nous allons l'ignorer pour l'instant car nous avons
# besoin de demander au client davantage de données.)
# Notons que nous devons inverser la transformation log10 effectuée lors de l'étape de nettoyage des données.
test_df['basecase_revenue'] = ((np.power(10, test_df['forecast_cons_12m']) + 1) * test_df['forecast_price_energy_off_peak']
                               + test_df['forecast_meter_rent_12m'])

# En tenant compte du churn
test_df['basecase_revenue_after_churn'] = test_df['basecase_revenue'] * (1 - 0.919 * test_df['churn'])
df_revenu = test_df.head()

def get_rev_delta(pred: pd.DataFrame, cutoff: float=0.5, discount: float=0.2) -> float:
    """
    Get the delta of revenues for offering discount for all customers with predicted churn risk >= cutoff
    """
    pred['discount_revenue'] = pred['basecase_revenue_after_churn']
    # Churn predicted => discount is given => customer stays for full year, independent of whether the prediction
    # (false positive, "free"/unnecessary discount given) or correct
    pred.loc[pred['churn_probability'] >= cutoff, 'discount_revenue'] = pred['basecase_revenue'] * (1 - discount)
    # Save the revenue delta for each customer in a separate column
    pred['revenue_delta'] = pred['discount_revenue'] - pred['basecase_revenue_after_churn']
    return pred['revenue_delta'].sum()

# Generate a list of possible cutoffs and the corresponding overall revenue deltas
rev_deltas = pd.Series({cutoff: get_rev_delta(test_df, cutoff=cutoff) for cutoff in np.arange(0, 1, 0.01)})

def plot_tradeoff(rev_deltas: pd.Series):
    # Plot the revenue deltas
    fig, ax = plt.subplots()
    rev_deltas.plot(ax=ax)
    # Mark optimal point
    max_pred = rev_deltas.idxmax()
    ax.scatter(max_pred, rev_deltas.loc[max_pred], s=100, c='red')
    # Reference line for break-even
    ax.hlines(0, 0, 1)
    plt.show()
    print(f'Maximum benefit at cutoff {max_pred} with revenue delta of ${rev_deltas.loc[max_pred]:,.2f}')
    st.pyplot(fig)

# comment choisir le seuil de coupure ?

def get_rev_delta_high_value(pred: pd.DataFrame, cutoff: float=0.5, discount: float=0.2, min_rev: float=500):
    """
    Get the delta of revenues for offering discount for all customers with predicted churn risk >= cutoff and rev
    """
    pred['discount_revenue'] = pred['basecase_revenue_after_churn']
    # Churn predicted => discount is given for high-value customers => customer stays for full year, independent
    # (false positive, "free"/unnecessary discount given) or correct
    pred.loc[(pred['churn_probability'] >= cutoff) & (pred['basecase_revenue'] > min_rev),
    'discount_revenue'] = pred['basecase_revenue'] * (1 - discount)
    # Save the revenue delta for each customer in a separate column
    pred['revenue_delta'] = pred['discount_revenue'] - pred['basecase_revenue_after_churn']
    return pred['revenue_delta'].sum()

# Générons une liste de seuils possibles et les deltas de revenus globaux correspondants
rev_deltas_high_value = pd.Series({cutoff: get_rev_delta_high_value(test_df, cutoff=cutoff) for cutoff in np.arange(0, 1, 0.01)})

# Les revenus d'électricité pour chaque client comprennent la consommation d'énergie (quantité * prix) et la location
# du compteur.
# Le prix de l'électricité peut également jouer un rôle, mais nous allons l'ignorer pour l'instant car nous devons
# demander plus d'informations au client.
# Notons que nous devons inverser la transformation logarithmique en base 10 effectuée lors de l'étape de nettoyage
# des données.
test_df['basecase_revenue'] = (np.power(10, test_df['forecast_cons_12m']) * test_df['forecast_price_energy_off_peak']
                               + test_df['forecast_meter_rent_12m'])
# Taking churn into account
test_df['basecase_revenue_after_churn'] = test_df['basecase_revenue'] * (1 - 0.919 * test_df['churn_probability'])

# Generate a list of possible cutoffs and the corresponding overall revenue deltas
rev_deltas = pd.Series({cutoff: get_rev_delta(test_df, cutoff=cutoff) for cutoff in np.arange(0, 1, 0.01)})





st.sidebar.title("Menu de Navigation")
selected_option = st.sidebar.radio("Plan de travail", ["Comprehension du besoin client",
                                                       "Presentation du jeu de données", "Exploration des données",
                                                       "Exploration des Hypothèses", "Feature Engineering",
                                                       "Modélisation", "Impact Business de la Remise de 20%",
                                                       "Conclusion et Recommandations"])

if selected_option == "Comprehension du besoin client":
    st.subheader("Contexte ")
    st.markdown("""
    PowerCo est un important fournisseur de gaz et d'électricité qui approvisionne des entreprises, des PME et des
     clients résidentiels. La libéralisation du marché de l'énergie en Europe a entraîné un taux de désabonnement 
     important, en particulier dans le segment des PME.
     
     Elle cherche à comprendre les raisons derrière le taux significatif de désabonnement dans le segment des PME.
     
     l'entreprise explore les hypothèses suivantes :
     
     - Les variations de prix ont un impact direct sur la perte de clients
     - Pour les clients qui risquent de changer de fournisseur, une remise  de 20 % pourrait les inciter à rester
      chez nous. """)
    st.subheader("Méthodologie de résolution")
    st.markdown("""
    Afin de tester l'hypothèse selon laquelle le désabonnement est lié à la sensibilité des clients aux prix, nous 
    devons modéliser les probabilités de désabonnement des clients et calculer l'effet des prix sur les taux de 
    désabonnement. Nous aurons besoin des données suivantes pour pouvoir construire les modèles.
    
**Données nécessaires** :
- Données sur les clients - qui devraient inclure les caractéristiques de chaque client, par exemple,
le secteur d'activité, l'historique de la consommation d'électricité, la date à laquelle le client est devenu client,
 etc.
- Données sur le désabonnement - qui doivent indiquer si le client s'est désabonné.
- Données historiques sur les prix - qui doivent indiquer les prix facturés par le client à chaque client,  tant pour 
l'électricité que pour le gaz, au moment de l'achat.

Une fois que nous aurons les données, le plan de travail sera le suivant
1. Nous devrions définir ce qu'est la sensibilité aux prix et la calculer.
2. Nous devrions concevoir des caractéristiques basées sur les données que nous obtenons, et construire un modèle de 
classification binaire (par exemple, régression logistique, Random Forest, Gradient Boosting).
3. Le meilleur modèle sera choisi en fonction du compromis entre la complexité, l'explicabilité et la précision
 des modèles.
4. Nous approfondirons ensuite la question de savoir pourquoi et comment les changements de prix ont un impact sur 
le taux de désabonnement.
5. Enfin, le modèle nous permettrait d'évaluer l'impact commercial de la stratégie d'actualisation proposée par 
le client.
    """)
    st.markdown(""" On y va !!! """)

if selected_option == "Presentation du jeu de données":
    st.markdown(""" Nous disposons d'un ensemble de données comprenant les caractéristiques des PME clientes en janvier
     2016 ainsi que des informations indiquant si elles se sont désabonnées ou non en mars 2016. 
     En outre, nous avons reçu les prix de 2015 pour ces clients.""")

    afficher_paragraphe2 = st.checkbox("Jeu de données")
    if afficher_paragraphe2:
        st.dataframe(client_df.head(3))
        st.markdown(""" Les données relatives aux clients sont un mélange de données numériques et catégorielles, 
        que nous devrons transformer avant de les modéliser ultérieurement """)
        st.dataframe(price_df.head(3))
        st.markdown("""En ce qui concerne les données relatives aux prix, il s'agit de données essentiellement
         numériques, mais nous pouvons constater la présence de valeurs nulles """)

    stat_desc = st.checkbox("Statistiques descriptives")
    if stat_desc:
        st.write("#### Informations sur les types de données:")
        display_data_types_info(client_df)
        display_data_types_info(price_df)
        st.write("Concernant les données client, nous constatons que toutes les variables liées à la date ne sont pas actuellement au format datetime."
                 " Nous devrons les convertir ultérieurement.")
        st.write("#### Statistics")
        st.dataframe(client_df.describe())
        st.markdown(""" Le point essentiel à retenir est que nous avons des données fortement asymétriques, comme le 
        montrent les valeurs des percentiles.""")
        st.dataframe(price_df.describe())
        st.markdown("Dans l'ensemble, les données relatives aux prix sont satisfaisantes.")

if selected_option == "Exploration des données" :
    st.write("Voyons maintenant un peu plus en détail les données clients et prix.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Taux d'attrition")
        plot_stacked_bars(churn_percentage.transpose(), "taux d'attrition", (5, 5), legend_="lower right")
    with col2:
        st.subheader("Canal de vente")
        plot_stacked_bars(channel_churn, 'Canal de vente', rot_=30)
        st.write(""" Les clients qui résilient sont répartis sur 5 valeurs différentes pour le canal de vente. De plus,
         la valeur "MISSING" a un taux d'attrition de 7,6 %. "MISSING" indique une valeur manquante. Cette 
         caractéristique pourrait être importante lors de la construction de notre modèle.""")

    st.subheader("Consommation")
    st.write("""
Voyons la distribution de la consommation au cours de la dernière année et du dernier mois. Comme les données de
 consommation sont univariées, utilisons des histogrammes pour visualiser leur distribution.""")
    consumption = client_df[['id', 'cons_12m', 'cons_gas_12m', 'cons_last_month', 'imp_cons', 'has_gas', 'churn']]
    variable_list = ['cons_12m', 'cons_gas_12m', 'cons_last_month', 'imp_cons']
    selected_variable = st.selectbox("Sélectionnez la variable à afficher :", variable_list)
    if selected_variable == "cons_gas_12m":
        plot_distribution(consumption[consumption['has_gas'] == 't'], selected_variable)

    else:
        plot_distribution(consumption, selected_variable)

    st.write(""" les données de consommation présentent une forte asymétrie positive, avec une très longue queue droite
     vers les valeurs plus élevées de la distribution. Les valeurs à l'extrémité supérieure et inférieure de la 
     distribution sont susceptibles d'être des valeurs aberrantes. 
     
     Nous pouvons utiliser un graphique standard pour visualiser les valeurs aberrantes plus en détail. 
     Une boîte à moustaches (boxplot) est une méthode standardisée d'affichage de la distribution basée sur un
      résumé en cinq nombres :
      - Minimum
      - Premier quartile (Q1)
      - Médiane
      - Troisième quartile (Q3)
      - Maximum
      
      Il peut révéler les outliers et quelles sont leurs valeurs. Il peut également nous indiquer si nos données 
      sont symétriques, à quel point nos données sont regroupées et si/comment nos données sont asymétriques.
        """)
    fig, axs = plt.subplots(nrows=4, figsize=(8, 10))

    # Plot histogram
    sns.boxplot(consumption["cons_12m"], orient="h", ax=axs[0])
    sns.boxplot(consumption[consumption["has_gas"] == "t"]["cons_gas_12m"], orient="h", ax=axs[1])
    sns.boxplot(consumption["cons_last_month"], orient="h", ax=axs[2])
    sns.boxplot(consumption["imp_cons"], orient="h", ax=axs[3])

    axs[0].set_xlim(-200000, 2000000)
    axs[1].set_xlim(-200000, 2000000)
    axs[2].set_xlim(-20000, 100000)
    st.pyplot(fig)
    st.subheader("Prévision")
    variable_forecast = ["forecast_cons_12m", "forecast_cons_year", "forecast_discount_energy",
                         "forecast_meter_rent_12m",
                         "forecast_price_energy_off_peak", "forecast_price_energy_peak", "forecast_price_pow_off_peak"]
    selected_var_forecast = st.selectbox("Sélectionnez la variable à afficher :", variable_forecast)
    plot_distribution(client_df, selected_var_forecast)
    st.write(""" De manière similaire aux graphiques de consommation, nous pouvons observer que de nombreuses variables
     présentent une forte asymétrie positive, créant une queue très longue pour les valeurs plus élevées. 
     Nous comptons effectuer certaines transformations pour corriger cette asymétrie.""")

    st.subheader("Type de contrat")
    contract = contract_type.groupby([contract_type['churn'], contract_type['has_gas']])['id'].count().unstack(level=0)
    contract_percentage = (contract.div(contract.sum(axis=1), axis=0) * 100).sort_values(by=[1], ascending=False)
    plot_stacked_bars(contract_percentage, 'Type de contrat avec le gaz')

    st.subheader("Marges")
    fig, axs = plt.subplots(nrows=3, figsize=(18, 20))
    # Plot histogram
    sns.boxplot(margin["margin_gross_pow_ele"], orient="h", ax=axs[0])
    sns.boxplot(margin["margin_net_pow_ele"], orient="h", ax=axs[1])
    sns.boxplot(margin["net_margin"], orient="h", ax=axs[2])
    st.pyplot(fig)
    st.write("""Nous pouvons également observer quelques valeurs aberrantes ici, que nous traiterons par la suite.""")
    st.subheader("L'énergie souscrite")
    plot_distribution(power, 'pow_max')

    st.subheader("Le reste des variables")
    others = client_df[['id', 'nb_prod_act', 'num_years_antig', 'origin_up', 'churn']]
    variable_list_others = ['nb_prod_act', 'num_years_antig', 'origin_up']
    selected_other_var = st.selectbox("Sélectionnez la variable à afficher :", variable_list_others)

    if selected_other_var == 'nb_prod_act':
        products = others.groupby([others[selected_other_var], others["churn"]])["id"].count().unstack(level=1)
        products_percentage = (products.div(products.sum(axis=1), axis=0) * 100).sort_values(by=[1], ascending=False)
        plot_stacked_bars(products_percentage, "Number of products")
    elif selected_other_var == 'num_years_antig':
        years_antig = others.groupby([others[selected_other_var], others["churn"]])["id"].count().unstack(level=1)
        years_antig_percentage = (years_antig.div(years_antig.sum(axis=1), axis=0) * 100)
        plot_stacked_bars(years_antig_percentage, "Number years")
    else:
        origin = others.groupby([others[selected_other_var], others["churn"]])["id"].count().unstack(level=1)
        origin_percentage = (origin.div(origin.sum(axis=1), axis=0) * 100)
        plot_stacked_bars(origin_percentage, "Origin contract/offer")


if selected_option == "Exploration des Hypothèses":
    st.write("""Maintenant que nous avons exploré les données, il est temps d'investiguer si la sensibilité au prix a 
    une influence sur le désabonnement. Tout d'abord, nous devons définir précisément ce qu'est la sensibilité au prix.
    
    Étant donné que nous disposons des données de consommation pour chacune des entreprises pour l'année 2015, 
    nous allons créer de nouvelles caractéristiques pour mesurer la "sensibilité au prix" en utilisant la moyenne
     de l'année, des 6 derniers mois et des 3 derniers mois.
        """)
    transform_price = st.checkbox("Caractéristiques de la sensibilité au prix (cochez cette petite case pour"
                                  " comprendre la creation des nouvelles variables associées)")
    if transform_price:
        st.subheader("Transformer les colonnes de date en type datetime")
        code_datetime = """
            client_df["date_activ"] = pd.to_datetime(client_df["date_activ"], format='%Y-%m-%d')
            client_df["date_end"] = pd.to_datetime(client_df["date_end"], format='%Y-%m-%d')
            client_df["date_modif_prod"] = pd.to_datetime(client_df["date_modif_prod"], format='%Y-%m-%d')
            client_df["date_renewal"] = pd.to_datetime(client_df["date_renewal"], format='%Y-%m-%d')
            price_df['price_date'] = pd.to_datetime(price_df['price_date'], format='%Y-%m-%d')
            """
        st.code(code_datetime, language="python")

        st.subheader("Créer des données de moyenne pondérée")
        code_mean = """
                mean_year = price_df.groupby(['id']).mean().reset_index()
                mean_6m = price_df[price_df['price_date'] > '2015-06-01'].groupby(['id']).mean().reset_index()
                mean_3m = price_df[price_df['price_date'] > '2015-10-01'].groupby(['id']).mean().reset_index()

                # Combiner en un seul dataframe
                mean_year = mean_year.rename(
                    index=str,
                    columns={
                        "price_off_peak_var": "mean_year_price_off_peak_var",
                        "price_peak_var": "mean_year_price_peak_var",
                        "price_mid_peak_var": "mean_year_price_mid_peak_var",
                        "price_off_peak_fix": "mean_year_price_off_peak_fix",
                        "price_peak_fix": "mean_year_price_peak_fix",
                        "price_mid_peak_fix": "mean_year_price_mid_peak_fix"
                    }
                )

                mean_year["mean_year_price_off_peak"] = mean_year["mean_year_price_off_peak_var"] + mean_year["mean_year_price_off_peak_fix"]
                mean_year["mean_year_price_peak"] = mean_year["mean_year_price_peak_var"] + mean_year["mean_year_price_peak_fix"]
                mean_year["mean_year_price_mid_peak"] = mean_year["mean_year_price_mid_peak_var"] + mean_year["mean_year_price_mid_peak_fix"]

                mean_6m = mean_6m.rename(
                    index=str,
                    columns={
                        "price_off_peak_var": "mean_6m_price_off_peak_var",
                        "price_peak_var": "mean_6m_price_peak_var",
                        "price_mid_peak_var": "mean_6m_price_mid_peak_var",
                        "price_off_peak_fix": "mean_6m_price_off_peak_fix",
                        "price_peak_fix": "mean_6m_price_peak_fix",
                        "price_mid_peak_fix": "mean_6m_price_mid_peak_fix"
                    }
                )
                mean_6m["mean_6m_price_off_peak"] = mean_6m["mean_6m_price_off_peak_var"] + mean_6m["mean_6m_price_off_peak_fix"]
                mean_6m["mean_6m_price_peak"] = mean_6m["mean_6m_price_peak_var"] + mean_6m["mean_6m_price_peak_fix"]
                mean_6m["mean_6m_price_mid_peak"] = mean_6m["mean_6m_price_mid_peak_var"] + mean_6m["mean_6m_price_mid_peak_fix"]

                mean_3m = mean_3m.rename(
                    index=str,
                    columns={
                        "price_off_peak_var": "mean_3m_price_off_peak_var",
                        "price_peak_var": "mean_3m_price_peak_var",
                        "price_mid_peak_var": "mean_3m_price_mid_peak_var",
                        "price_off_peak_fix": "mean_3m_price_off_peak_fix",
                        "price_peak_fix": "mean_3m_price_peak_fix",
                        "price_mid_peak_fix": "mean_3m_price_mid_peak_fix"
                    }
                )
                mean_3m["mean_3m_price_off_peak"] = mean_3m["mean_3m_price_off_peak_var"] + mean_3m["mean_3m_price_off_peak_fix"]
                mean_3m["mean_3m_price_peak"] = mean_3m["mean_3m_price_peak_var"] + mean_3m["mean_3m_price_peak_fix"]
                mean_3m["mean_3m_price_mid_peak"] = mean_3m["mean_3m_price_mid_peak_var"] + mean_3m["mean_3m_price_mid_peak_fix"]

                # Fusionner en seul dataframe
                price_features = pd.merge(mean_year, mean_6m, on='id')
                price_features = pd.merge(price_features, mean_3m, on='id')

                    """
        st.code(code_mean, language="python")

    st.dataframe(price_features.head())

    st.write("Maintenant, fusionnons les données de churn et voyons s'il y a une corrélation avec la sensibilité "
             "au prix.")

    corr = price_analysis_copy.corr()
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True,
                annot_kws={'size': 10})

    # Ajustement de la taille des étiquettes d'axe
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Affichage du graphique dans Streamlit en passant explicitement la figure
    st.pyplot(fig)
    st.write(""" D'après le graphique de corrélation, il montre une plus grande intensité de corrélation entre les
     autres variables de sensibilité au prix, cependant, dans l'ensemble, la corrélation avec le désabonnement est
      très faible. Cela indique qu'il existe une faible relation linéaire entre la sensibilité au prix et 
      le désabonnement. Cela suggère que, pour que la sensibilité au prix soit un facteur majeur dans la prédiction
       du taux d'attrition, nous devrons peut-être modifier la transformation des caracteristiques correspondantes.""")

    st.dataframe(merged_data.head(3))
    num_rows, num_columns = merged_data.shape
    st.write(f"La base de données nettoyés a {num_rows} lignes et {num_columns} colonnes.")
    # st.text(merged_data.columns)

if selected_option == "Feature Engineering":
    st.subheader("Différence entre les prix hors pointe en décembre et janvier précédent")
    st.write("""Ci-dessous est le code créé pour calculer la caractéristique décrite ci-dessus.""")
    # st.dataframe(price_df.head(3))
    st.write(""" Ici, nous :
    - Regroupons les prix hors pointe par entreprise et par mois
    - Obtenons les prix de janvier et de décembre
    - Calculons la différence
    
    Nous obtenons """)

    st.dataframe(diff.head(3))
    st.subheader("Variation moyenne des prix sur les périodes")
    st.write("""Nous pouvons maintenant améliorer la caractéristique  en calculant la 
    variation moyenne des prix sur des périodes individuelles, plutôt que sur l'ensemble de l'année.""")
    st.write(""" Ici, nous :
    - Agrégeons les prix moyens par période par entreprise
    - Calculons la différence moyenne entre les périodes consécutives : 'off_peak_peak_var_mean_diff', 
     'peak_mid_peak_var_mean_diff',  'off_peak_mid_peak_var_mean_diff',  'off_peak_peak_fix_mean_diff', 
     'peak_mid_peak_fix_mean_diff', 'off_peak_mid_peak_fix_mean_diff'
      """)
    code_agg = """# Aggregate average prices per period by company
    mean_prices = price_df.groupby(['id']).agg({
        'price_off_peak_var': 'mean',
        'price_peak_var': 'mean',
        'price_mid_peak_var': 'mean',
        'price_off_peak_fix': 'mean',
        'price_peak_fix': 'mean',
        'price_mid_peak_fix': 'mean'
    }).reset_index()

    # Calculate the mean difference between consecutive periods
    mean_prices['off_peak_peak_var_mean_diff'] = mean_prices['price_off_peak_var'] - mean_prices['price_peak_var']
    mean_prices['peak_mid_peak_var_mean_diff'] = mean_prices['price_peak_var'] - mean_prices['price_mid_peak_var']
    mean_prices['off_peak_mid_peak_var_mean_diff'] = mean_prices['price_off_peak_var'] - mean_prices[
        'price_mid_peak_var']
    mean_prices['off_peak_peak_fix_mean_diff'] = mean_prices['price_off_peak_fix'] - mean_prices['price_peak_fix']
    mean_prices['peak_mid_peak_fix_mean_diff'] = mean_prices['price_peak_fix'] - mean_prices['price_mid_peak_fix']
    mean_prices['off_peak_mid_peak_fix_mean_diff'] = mean_prices['price_off_peak_fix'] - mean_prices[
        'price_mid_peak_fix']
    columns = [
        'id',
        'off_peak_peak_var_mean_diff',
        'peak_mid_peak_var_mean_diff',
        'off_peak_mid_peak_var_mean_diff',
        'off_peak_peak_fix_mean_diff',
        'peak_mid_peak_fix_mean_diff',
        'off_peak_mid_peak_fix_mean_diff'
    ]
    df = pd.merge(df, mean_prices[columns], on='id')"""
    st.code(code_agg, language="python")

    st.write(""" Ces variables peuvent etre utiles, car elles ajoutent plus d'intensité aux variables de variations de
     prix crées précedement. Au lieu de regarder les différences sur une année entière, nous avons maintenant créé
      des variables qui examinent les différences moyennes de prix sur différentes périodes
       (hors pointe, pointe, mi-pointe). La fonction déc-jan peut révéler des tendances macro qui se produisent sur
        une année entière, tandis que les variables inter-périodes peuvent révéler des tendances à une échelle micro 
        entre les mois.""")

    st.subheader("Changements de prix maximaux à travers les périodes et les mois")
    st.write(""" Une autre façon dont nous pouvons améliorer la caractéristique  est d'examiner 
    le changement maximal de prix à travers les périodes et les mois.
     Nous :
     - Agrégeons les prix moyens par période par entreprise
     - Calculons la différence moyenne entre les périodes consécutives
     - Calculons la différence mensuelle maximale à travers les périodes de temps
     """)

    code_price = """# Aggregate average prices per period by company
    mean_prices_by_month = price_df.groupby(['id', 'price_date']).agg({
        'price_off_peak_var': 'mean',
        'price_peak_var': 'mean',
        'price_mid_peak_var': 'mean',
        'price_off_peak_fix': 'mean',
        'price_peak_fix': 'mean',
        'price_mid_peak_fix': 'mean'
    }).reset_index()

    # Calculate the mean difference between consecutive periods
    mean_prices_by_month['off_peak_peak_var_mean_diff'] = mean_prices_by_month['price_off_peak_var'] - \
                                                          mean_prices_by_month['price_peak_var']
    mean_prices_by_month['peak_mid_peak_var_mean_diff'] = mean_prices_by_month['price_peak_var'] - mean_prices_by_month[
        'price_mid_peak_var']
    mean_prices_by_month['off_peak_mid_peak_var_mean_diff'] = mean_prices_by_month['price_off_peak_var'] - \
                                                              mean_prices_by_month['price_mid_peak_var']
    mean_prices_by_month['off_peak_peak_fix_mean_diff'] = mean_prices_by_month['price_off_peak_fix'] - \
                                                          mean_prices_by_month['price_peak_fix']
    mean_prices_by_month['peak_mid_peak_fix_mean_diff'] = mean_prices_by_month['price_peak_fix'] - mean_prices_by_month[
        'price_mid_peak_fix']
    mean_prices_by_month['off_peak_mid_peak_fix_mean_diff'] = mean_prices_by_month['price_off_peak_fix'] - \
                                                              mean_prices_by_month['price_mid_peak_fix']
    # Calculate the maximum monthly difference across time periods
    max_diff_across_periods_months = mean_prices_by_month.groupby(['id']).agg({
        'off_peak_peak_var_mean_diff': 'max',
        'peak_mid_peak_var_mean_diff': 'max',
        'off_peak_mid_peak_var_mean_diff': 'max',
        'off_peak_peak_fix_mean_diff': 'max',
        'peak_mid_peak_fix_mean_diff': 'max',
        'off_peak_mid_peak_fix_mean_diff': 'max'
    }).reset_index().rename(
        columns={
            'off_peak_peak_var_mean_diff': 'off_peak_peak_var_max_monthly_diff',
            'peak_mid_peak_var_mean_diff': 'peak_mid_peak_var_max_monthly_diff',
            'off_peak_mid_peak_var_mean_diff': 'off_peak_mid_peak_var_max_monthly_diff',
            'off_peak_peak_fix_mean_diff': 'off_peak_peak_fix_max_monthly_diff',
            'peak_mid_peak_fix_mean_diff': 'peak_mid_peak_fix_max_monthly_diff',
            'off_peak_mid_peak_fix_mean_diff': 'off_peak_mid_peak_fix_max_monthly_diff'
        }
    )
    columns = [
        'id',
        'off_peak_peak_var_max_monthly_diff',
        'peak_mid_peak_var_max_monthly_diff',
        'off_peak_mid_peak_var_max_monthly_diff',
        'off_peak_peak_fix_max_monthly_diff',
        'peak_mid_peak_fix_max_monthly_diff',
        'off_peak_mid_peak_fix_max_monthly_diff'
    ]

    df = pd.merge(df, max_diff_across_periods_months[columns], on='id')"""
    st.code(code_price, language="python")

    st.write(""" Nous avons pener que calculer la variation maximale des prix entre les mois et les périodes serait
     une bonne caractéristique à créer, car j'essayais de réfléchir du point de vue d'un client de PowerCo. 
     En tant que client des services publics, rien n'est plus agaçant que des changements soudains de prix entre les
      mois, et une forte augmentation des prix sur une courte période serait un facteur influent me poussant à examiner
       d'autres fournisseurs de services publics pour une meilleure offre. Puisque nous essayons de prédire le 
       désabonnement pour ce cas d'utilisation, j'ai pensé que ce serait une caractéristique intéressante 
       à inclure. """)

    st.subheader("Transformations des autres variables assymetriques  ")
    st.write(""" Cette section aborde la transformation des variables supplémentaires auxquelles nous avons peut-être
     pensé, ainsi que différentes façons de transformer nos données pour prendre en compte certaines de leurs 
     propriétés statistiques que nous avons vues précédemment, telles que la symétrie.""")
    st.markdown("##### Tenure")
    st.write(""" Pendant combien de temps une entreprise est-elle cliente de PowerCo ?""")
    st.dataframe(df.groupby(['tenure']).agg({'churn': 'mean'}).sort_values(by='churn', ascending=False))
    st.write(""" Nous pouvons voir que les entreprises qui sont clientes depuis seulement 4 mois ou moins sont beaucoup
     plus susceptibles de se désabonner par rapport aux entreprises qui sont clientes depuis plus longtemps.
      Et donc, la différence entre 4 et 5 mois est d'environ 4 %, ce qui représente un saut important dans la 
      probabilité qu'un client se désabonne par rapport aux autres différences entre les valeurs ordonnées de la durée.
       Cela révèle peut-être que le fait d'amener un client au-delà de 4 mois de durée est en réalité une étape 
       importante pour le maintenir en tant que client à long terme.
       
       C'est une caractéristique intéressante à conserver pour la modélisation car il est clair que la durée pendant 
       laquelle vous êtes client influence la probabilité de désabonnement du client.""")

    st.markdown(" ##### Transformer les dates en mois ")
    st.write(""""
   -  months_activ = Nombre de mois actifs jusqu'à la date de référence (Jan 2016)
   -  months_to_end = Nombre de mois de contrat restants jusqu'à la date de référence (Jan 2016)
   -  months_modif_prod = Nombre de mois depuis la dernière modification jusqu'à la date de référence (Jan 2016)
   -  months_renewal = Nombre de mois depuis le dernier renouvellement jusqu'à la date de référence (Jan 2016)""")


    code_months = """# Create reference date
    reference_date = datetime(2016, 1, 1)

    # Create columns
    df['months_activ'] = convert_months(reference_date, df, 'date_activ')
    df['months_to_end'] = -convert_months(reference_date, df, 'date_end')
    df['months_modif_prod'] = convert_months(reference_date, df, 'date_modif_prod')
    df['months_renewal'] = convert_months(reference_date, df, 'date_renewal')"""
    st.code(code_months, language="python")

    st.write(""" Les dates sous forme datetime ne sont pas utiles pour un modèle prédictif, nous devions donc utiliser
     ces dates pour créer d'autres caractéristiques qui pourraient avoir une certaine puissance prédictive.
      En utilisant l'intuition, on pourrait supposer qu'un client qui est client actif de PowerCo depuis plus 
      longtemps pourrait avoir plus de fidélité à la marque et est plus susceptible de rester. 
      Alors qu'un client plus récent pourrait être plus volatile. D'où l'ajout de la caractéristique months_activ.

      De plus, si nous pensons du point de vue d'un client de PowerCo, si vous approchez de la fin de votre contrat
       avec PowerCo, vos pensées pourraient aller dans quelques directions. 
       Vous pourriez rechercher de meilleures offres pour quand votre contrat se termine, ou vous pourriez vouloir 
       voir votre contrat actuel jusqu'à son terme et en signer un autre. D'un autre côté, si vous venez de vous 
       joindre, vous pourriez avoir une période où vous êtes autorisé à partir si vous n'êtes pas satisfait. 
       De plus, si vous êtes au milieu de votre contrat, il pourrait y avoir des frais si vous vouliez partir, 
       dissuadant les clients de se désabonner au milieu de leur accord. Ainsi, je pense que months_to_end sera une 
       caractéristique intéressante car elle peut révéler des schémas et des comportements concernant le moment
        du désabonnement.

        Je crois que si un client a apporté des mises à jour récentes à son contrat, il est plus susceptible d'être
         satisfait ou du moins il a reçu un niveau de service client pour mettre à jour ou modifier ses services
          existants. Je crois que cela est un signe positif, montrant qu'ils sont un client engagé, et donc je pense 
          que months_modif_prod sera une caractéristique intéressante à inclure car elle montre le degré d'engagement 
          d'un client avec PowerCo.

     Enfin, le nombre de mois depuis la dernière fois qu'un client a renouvelé un contrat sera, à mon avis, 
     une caractéristique intéressante car une fois de plus, elle montre le degré d'engagement de ce client. Cela va 
     également plus loin que l'engagement, montrant un niveau d'engagement si un client renouvelle son contrat.
      Pour cette raison, je pense que months_renewal sera une bonne caractéristique à inclure. """)
    st.write(""" Nous n'avons plus besoin des colonnes datetime que nous avons utilisées pour les transformations,
     nous pouvons donc les supprimer.""")
    code_remove = """remove = [
        'date_activ',
        'date_end',
        'date_modif_prod',
        'date_renewal'
    ]

    df = df.drop(columns=remove)"""
    st.code(code_remove, language="python")
    st.subheader("Transformation des données booléennes")
    st.write("has_gas")
    st.write("Nous voulons simplement transformer cette colonne de catégorique en un indicateur binaire.")
    st.dataframe(df.groupby(['has_gas']).agg({'churn': 'mean'}))
    st.write(""" Si un client achète également du gaz chez PowerCo, cela montre qu'il possède plusieurs produits 
    et est un client fidèle à la marque. Il n'est donc pas surprenant que les clients qui n'achètent pas de gaz 
    soient presque 2 % plus susceptibles de résilier leur contrat que les clients qui achètent également du gaz 
    chez PowerCo. C'est donc une caractéristique utile.""")
    st.subheader("Transformer les données catégorielles")
    st.markdown(""" Un modèle prédictif ne peut pas accepter des valeurs catégorielles ou de chaîne, donc en tant que
     data scientist, nous devons encoder les caractéristiques catégorielles en représentations numériques de la
      manière la plus compacte et discriminative possible. La méthode la plus simple consiste à mapper chaque catégorie
       sur un entier (encodage par étiquette), cependant cela n'est pas toujours approprié car cela introduit ensuite
        le concept d'ordre dans une caractéristique qui peut ne pas être intrinsèquement présente 0 < 1 < 2 < 3 ...

      Une autre façon d'encoder les caractéristiques catégorielles est d'utiliser des variables factices,
       également appelées encodage one-hot. Cela crée une nouvelle caractéristique pour chaque valeur unique d'une 
       colonne catégorielle et remplit cette colonne avec un 1 ou un 0 pour indiquer si cette entreprise appartient
        ou non à cette catégorie..""")
    st.markdown("##### channel_sales")

    # Let's see how many categories are within this column
    st.dataframe(client_df['channel_sales'].value_counts())
    st.write("""
    Nous avons 8 catégories, nous allons donc créer 8 variables factices à partir de cette colonne. Cependant, comme
     nous pouvons le voir pour les trois dernières catégories dans la sortie ci-dessus, elles ont respectivement 
     11, 3 et 2 occurrences. Étant donné que notre ensemble de données compte environ 14 000 lignes,
      cela signifie que ces variables factices seront presque entièrement égales à 0 et n'ajouteront 
      donc pas beaucoup de puissance prédictive au modèle (car elles sont presque entièrement constantes 
      et fournissent très peu d'information).
     Pour cette raison, nous allons supprimer ces 3 variables factices.""")

    code_dummies ="""df = pd.get_dummies(df, columns=['channel_sales'], prefix='channel')
    df = df.drop(columns=['channel_sddiedcslfslkckwlfkdpoeeailfpeds', 'channel_epumfxlbckeskwekxbiuasklxalciiuu',
                          'channel_fixdbufsefwooaasfcxdxadsiekoceaa'])"""
    st.code(code_dummies, language="python")
    st.markdown("##### origin_up")
    # Let's see how many categories are within this column
    st.dataframe(client_df['origin_up'].value_counts())
    st.write(""" De manière similaire à `channel_sales`, les trois dernières catégories dans la sortie ci-dessus
     montrent une fréquence très faible. Nous allons donc les supprimer des caractéristiques après la création 
     des variables factices. """)

    st.subheader("Transformation des données numériques")
    st.markdown(""" Dans la partie prédedente, nous avons constaté que certaines variables étaient fortement 
    asymétriques. La raison pour laquelle nous devons traiter l'asymétrie est que certains modèles prédictifs ont 
    des hypothèses inhérentes sur la distribution des caractéristiques qui leur sont fournies. 
    Ces modèles sont appelés modèles paramétriques et ils supposent généralement que toutes les variables sont à 
    la fois indépendantes et normalement distribuées.

L'asymétrie n'est pas toujours une mauvaise chose, mais en règle générale, il est toujours bon de traiter 
les variables fortement asymétriques pour les raisons mentionnées ci-dessus, mais aussi parce que cela peut
 améliorer la vitesse à laquelle les modèles prédictifs convergent vers leur meilleure solution.

Il existe de nombreuses façons de traiter les variables asymétriques. Nous pouvons appliquer des transformations
 telles que:
  - la racine carrée,
  - la racine cubique,
  - le logarithme à une colonne numérique continue.
  
   Pour ce projet, nous utiliserons la transformation du "logarithme" pour les caractéristiques positivement
    asymétriques.

    Note : Nous ne pouvons pas appliquer le logarithme à une valeur de 0, nous ajouterons donc une constante de 1 
    à toutes les valeurs.
    
    Tout d'abord, je veux voir les statistiques des caractéristiques asymétriques, afin que nous puissions
     comparer avant et après la transformation.""")


    st.dataframe(skewed_before)

    st.subheader("Nous pouvons constater que l'écart-type pour la plupart de ces caractéristiques est assez élevé.")
    # Apply log10 transformation
    st.dataframe(skewed_after)

    st.markdown("""
    Maintenant, nous pouvons voir que pour la majorité des caractéristiques, leur écart type est beaucoup plus bas 
    après la transformation. C'est une bonne chose, cela montre que ces caractéristiques sont maintenant plus 
    stables et prévisibles. Vérifions rapidement les distributions de certaines de ces caractéristiques également.""")

    col1, col2, col3 = st.columns(3)
    with col1:
        fig, ax = plt.subplots(figsize=(18, 20))
        sns.distplot((df["cons_12m"].dropna()))
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(18, 20))
        sns.distplot((df[df["has_gas"] == 1]["cons_gas_12m"].dropna()))
        st.pyplot(fig)

    with col3:
        fig, axs = plt.subplots(figsize=(18, 20))
        # Plot histograms
        sns.distplot((df["cons_last_month"].dropna()))
        st.pyplot(fig)

    st.subheader("Correlations")
    st.markdown("""
    En ce qui concerne la création de nouvelles variables et la transformation de celles existantes, c'est vraiment 
    une situation d'essais et d'erreurs qui nécessite de l'itération. Une fois que nous formons un modèle prédictif,
     nous pouvons voir quelles variables fonctionnent ou non, et nous saurons également à quel point cet ensemble
      de fonctionnalités est prédictif. Sur cette base, nous pouvons revenir à l'ingénierie des données pour 
      améliorer notre modèle.

      Pour l'instant, nous laisserons les l'ingénierie de données à ce stade. Une autre chose toujours utile à 
      examiner est la corrélation entre toutes les variables de votre ensemble de données.
     
     C'est important car cela révèle les relations linéaires entre les fonctionnalités. Nous voulons que
      les variables soient corrélées avec le désabonnement, car cela indiquera qu'elles sont de bons prédicteurs. 
      Cependant, des variables ayant une corrélation très élevée peuvent parfois être suspectes. 
      Cela est dû au fait que deux colonnes fortement corrélées indiquent qu'elles peuvent partager beaucoup des 
      mêmes informations. L'une des hypothèses de tout modèle prédictif paramétrique (comme indiqué précédemment) est 
      que toutes les variables doivent être indépendantes.

     Idéalement, nous voulons un ensemble de variables ayant une corrélation de 0 avec toutes les variables 
     indépendantes (toutes les variables sauf notre variable cible) et une corrélation élevée avec la variable 
     cible (désabonnement). Cependant, c'est très rarement le cas et il est courant d'avoir un faible degré de 
     corrélation entre les variables indépendantes.

     Regardons maintenant comment toutes les variables du modèle sont corrélées. """)

    df_copy = df.drop(['Unnamed: 0', 'id', 'price_date_x', 'price_date_y', 'price_date'], axis=1)
    # display_data_types_info(df)
    # st.dataframe(df.head(3))
    correlation = df_copy.corr()

    # Plot correlation

    fig, ax = plt.subplots(figsize=(45, 45))
    sns.heatmap(
        correlation,
        xticklabels=correlation.columns.values,
        yticklabels=correlation.columns.values,
        annot=True,
        annot_kws={'size': 12}
    )
    # Axis ticks size
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    st.pyplot(fig)

    st.write(""" Nous allons supprimer deux variables qui présentent une corrélation élevée avec d'autres
     variables indépendantes : 'num_years_antig' et 'forecast_cons_year'.""")

if selected_option == "Modélisation":

    st.write(""" Nous avons maintenant un ensemble de données contenant des caractéristiques que nous avons conçues et 
    nous sommes prêts à commencer l'entraînement d'un modèle prédictif. Nous nous concentrer uniquement
     sur l'entraînement d'un Random Forest. """)
    st.subheader("  L'échantillonnage de données ")
    st.write(""" La première chose que nous voulons faire est de diviser notre ensemble de données en échantillons
     d'entraînement et de test.""")
    code_model =""" 
    df_copy = df.drop(['Unnamed: 0', 'id', 'price_date_x', 'price_date_y', 'price_date'], axis=1)
    train_df = df_copy.copy()

    # Separate target variable from independent variables
    y = df['churn']
    X = df.drop(columns=['id', 'churn', 'Unnamed: 0', 'price_date_x', 'price_date_y', 'price_date'])
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)"""
    st.code(code_model, language="python")

    st.subheader("Model training")
    st.markdown("""
    Encore une fois, nous utilisons un classificateur de forêt aléatoire dans cet exemple. Une forêt aléatoire appartient à la catégorie des algorithmes d'ensemble car, en interne, la forêt fait référence à une collection d'arbres de décision qui sont des algorithmes d'apprentissage basés sur les arbres. En tant que data scientist, vous pouvez contrôler la taille de la forêt (c'est-à-dire le nombre d'arbres de décision que vous souhaitez inclure).

La raison pour laquelle un algorithme d'ensemble est puissant réside dans les lois de la moyenne, les apprenants faibles et le théorème central limite. Si nous prenons un seul arbre de décision et lui donnons un échantillon de données et certains paramètres, il apprendra des motifs à partir des données. Il peut être surajusté ou sous-ajusté, mais c'est maintenant notre seul espoir, ce modèle unique.

Avec les méthodes d'ensemble, au lieu de compter sur un seul modèle entraîné, nous pouvons former des milliers d'arbres de décision, tous utilisant des découpages différents des données et apprenant différents motifs. Cela reviendrait à demander à 1000 personnes d'apprendre toutes à coder. Vous finiriez avec 1000 personnes ayant différentes réponses, méthodes et styles ! La notion d'apprenant faible s'applique également ici. On a découvert que si vous entraînez vos apprenants à ne pas surajuster, mais à apprendre de faibles motifs dans les données et que vous avez beaucoup de ces apprenants faibles, ensemble, ils se réunissent pour former une piscine de connaissances hautement prédictive ! C'est une application de la vie réelle de la notion que plusieurs cerveaux valent mieux qu'un seul.

Maintenant, au lieu de compter sur un seul arbre de décision pour la prédiction, la forêt aléatoire le soumet aux opinions globales de l'ensemble complet des arbres de décision. Certains algorithmes d'ensemble utilisent une approche de vote pour décider de la meilleure prédiction, d'autres utilisent une moyenne.

À mesure que nous augmentons le nombre d'apprenants, l'idée est que les performances de la forêt aléatoire devraient converger vers sa meilleure solution possible.

Certains avantages supplémentaires du classificateur de forêt aléatoire incluent :

- La forêt aléatoire utilise une approche basée sur des règles au lieu d'un calcul de distance, et donc les fonctionnalités n'ont pas besoin d'être mises à l'échelle.
- Elle est capable de gérer les paramètres non linéaires mieux que les modèles basés sur la linéarité.
Cependant, certains inconvénients du classificateur de forêt aléatoire comprennent :

- La puissance de calcul nécessaire pour entraîner une forêt aléatoire sur un grand ensemble de données est élevée, car nous devons construire tout un ensemble d'estimateurs.
- Le temps d'entraînement peut être plus long en raison de la complexité et de la taille accrue de l'ensemble.
    """)
    code_model2= """
    model = RandomForestClassifier(
        n_estimators=1000
    )
    model.fit(X_train, y_train)"""
    st.code(code_model2, language="python")
    st.subheader("Evaluation")
    st.markdown(""" Évaluons maintenant à quel point ce modèle entraîné est capable de prédire les valeurs de l'ensemble de données de test.

Nous allons utiliser 3 métriques pour évaluer les performances :

- Précision = le rapport des observations correctement prédites sur l'ensemble des observations
- Précision = la capacité du classificateur à ne pas étiqueter un échantillon négatif comme positif
- Rappel = la capacité du classificateur à trouver tous les échantillons positifs

La raison pour laquelle nous utilisons ces trois métriques est que la simple précision n'est pas toujours une bonne mesure à utiliser. Pour donner un exemple, supposons que vous prédisez des insuffisances cardiaques chez des patients à l'hôpital et qu'il y avait 100 patients sur 1000 qui avaient une insuffisance cardiaque.

Si vous avez prédit correctement 80 sur 100 (80%) des patients qui avaient effectivement une insuffisance cardiaque, vous pourriez penser que vous avez bien fait ! Cependant, cela signifie également que vous avez prédit 20 erreurs et quelles pourraient être les implications de prédire ces 20 patients restants de manière incorrecte ? Peut-être qu'ils passent à côté d'un traitement vital pour sauver leur vie.

De plus, quelle est l'impact de prédire des cas négatifs comme positifs (personnes n'ayant pas une insuffisance cardiaque mais étant prédites positives) ? Peut-être qu'un grand nombre de faux positifs signifie que des ressources sont utilisées pour les mauvaises personnes et beaucoup de temps est gaspillé alors qu'ils auraient pu aider les véritables personnes souffrant d'insuffisance cardiaque.

Il s'agit simplement d'un exemple, mais il illustre pourquoi d'autres métriques de performance sont nécessaires, telles que la précision et le rappel, qui sont de bonnes mesures à utiliser dans un scénario de classification..""")


    st.write(f"True positives: {tp}")
    st.write(f"False positives: {fp}")
    st.write(f"True negatives: {tn}")
    st.write(f"False negatives: {fn}\n")

    st.write(f"Accuracy: {metrics.accuracy_score(y_test, predictions)}")
    st.write(f"Precision: {metrics.precision_score(y_test, predictions)}")
    st.write(f"Recall: {metrics.recall_score(y_test, predictions)}")

    st.markdown(""" En examinant ces résultats, quelques points à souligner :

Remarque : Si vous exécutez ce notebook vous-même, vous pourriez obtenir des réponses légèrement différentes !

Dans l'ensemble du jeu de test, environ 10% des lignes correspondent à des clients résiliants (churn = 1).
En ce qui concerne les vrais négatifs, nous en avons 3282 sur 3286. Cela signifie que sur l'ensemble des cas négatifs (churn = 0), nous avons prédit correctement 3282 comme négatifs (d'où le nom de vrais négatifs). C'est excellent !
En ce qui concerne les faux négatifs, cela correspond aux cas où nous avons prédit qu'un client ne résilierait pas (churn = 0) alors qu'en réalité il a résilié (churn = 1). Ce nombre est assez élevé à 348, nous voulons réduire autant que possible le nombre de faux négatifs, donc cela devrait être abordé lors de l'amélioration du modèle.
En ce qui concerne les faux positifs, cela correspond aux cas où nous avons prédit qu'un client résilierait alors qu'en réalité il n'a pas résilié. Pour cette valeur, nous pouvons voir qu'il y a 4 cas, ce qui est excellent !
Pour les vrais positifs, nous pouvons voir qu'au total, nous avons 366 clients qui ont résilié dans l'ensemble de données de test. Cependant, nous ne sommes capables d'identifier correctement que 18 de ces 366, ce qui est très faible.
En examinant le score de précision, il est très trompeur ! C'est pourquoi l'utilisation de la précision et du rappel est importante. Le score de précision est élevé, mais il ne nous dit pas toute l'histoire.
En examinant le score de rappel, cela montre un score de 0,82, ce qui n'est pas mal, mais pourrait être amélioré.
Cependant, le rappel montre que le classificateur a une très faible capacité à identifier les échantillons positifs. Cela serait la principale préoccupation pour l'amélioration de ce modèle !
En résumé, nous sommes capables d'identifier très précisément les clients qui ne résilient pas, mais nous ne sommes pas capables de prédire les cas où les clients résilient ! Ce que nous constatons, c'est qu'un pourcentage élevé de clients sont identifiés comme ne résiliant pas alors qu'ils devraient être identifiés comme résiliant. Cela me fait penser que l'ensemble actuel de fonctionnalités n'est pas assez discriminant pour distinguer clairement les résiliants des non-résiliants.

Un data scientist à ce stade reviendrait à l'ingénierie des fonctionnalités pour essayer de créer des fonctionnalités plus prédictives. Il pourrait également expérimenter avec l'optimisation des paramètres dans le modèle pour améliorer les performances. Pour l'instant, plongeons un peu plus dans la compréhension du modèle. """)
    st.subheader("Interpretation du modèle")
    st.markdown("""
    Une manière simple de comprendre les résultats d'un modèle est d'examiner les importances des fonctionnalités. 
    Les importances des fonctionnalités indiquent l'importance d'une fonctionnalité dans le modèle prédictif. 
    Il existe plusieurs façons de calculer l'importance des fonctionnalités, mais avec le classificateur Random Forest,
     nous sommes en mesure d'extraire les importances des fonctionnalités à l'aide de la méthode intégrée dans 
     le modèle entraîné. Dans le cas du Random Forest, l'importance des fonctionnalités représente le nombre de fois 
     que chaque fonctionnalité est utilisée pour diviser l'ensemble des arbres.
    """)

    fig, ax = plt.subplots(figsize=(15, 25))
    plt.title('Feature Importances')
    plt.barh(range(len(feature_importances)), feature_importances['importance'], color='b', align='center')
    plt.yticks(range(len(feature_importances)), feature_importances['features'])
    plt.xlabel('Importance')
    st.pyplot(fig)

    st.markdown(""" À partir de ce graphique, nous pouvons observer les points suivants :

- La marge nette et la consommation sur 12 mois sont des facteurs clés de l'attrition dans ce modèle.
- La marge sur l'abonnement d'électricité est également un facteur influent.
- Le temps semble être un facteur influent, en particulier le nombre de mois d'activité, la durée de leur relation contractuelle et le nombre de mois depuis leur dernière mise à jour de contrat.
- La caractéristique recommandée par notre collègue se situe dans la moitié supérieure en termes d'influence, et certaines des caractéristiques développées à partir de celle-ci la surpassent effectivement.
- Nos caractéristiques de sensibilité au prix sont dispersées, mais ne sont pas le principal moteur du départ d'un client.

La dernière observation est importante car elle renvoie à notre hypothèse initiale : """)

    st.subheader(" L'attrition est-elle motivée par la sensibilité au prix des clients ?")
    st.markdown(""" D'après les résultats des importances des fonctionnalités, ce n'est pas un facteur principal,
     mais plutôt un contributeur faible. Cependant, pour parvenir à une conclusion définitive, davantage 
     d'expérimentation est nécessaire.""")
    code_pred =""" proba_predictions = model.predict_proba(X_test)
    probabilities = proba_predictions[:, 1]
    X_test = X_test.reset_index()
    X_test.drop(columns='index', inplace=True)
    X_test['churn'] = predictions.tolist()
    X_test['churn_probability'] = probabilities.tolist()"""
    st.code(code_pred, language="python")

if selected_option == "Impact Business de la Remise de 20%" :
    st.markdown(""" Nous effectuons une analyse de la stratégie de remise proposée. L'entreprise a proposé d'offrir 
    une remise de 20% aux clients ayant une forte propension à la résiliation. Nous pouvons supposer, pour commencer,
     que tous ceux à qui une remise est offerte l'accepteront.""")
    st.markdown(""" Notre tâche est de calculer le chiffre d'affaires prévu pour l'ensemble des clients :
    - Lorsqu'aucune remise n'est proposée
    - Lorsqu'une remise est offerte en fonction d'un seuil de probabilité pour décider qui devrait recevoir la remise de 20%
    - Ainsi, décider où fixer le seuil pour maximiser le chiffre d'affaires.""")

    st.subheader("Calculons une estimation de revenus de base")
    st.markdown(" Calculons une estimation du revenu de l'électricité pour chaque client au cours des douze prochains"
                " mois, sur la base de la consommation prévue, du prix prévu et du taux de désabonnement réel."
                " Appelons cette estimation basecase_revenue.")

    st.markdown(""" Pour les clients qui finissent par résilier, nous devrions réduire notre calcul de chiffre 
    d'affaires prévu de 91,9 % pour tenir compte des clients résiliant entre janvier 2016 et le début de mars 2016. 
    (Ne sachant pas quand ils résilient, une hypothèse raisonnable pour la perte de revenus est la moyenne de 100 %,
     correspondant à la résiliation le 1er janvier 2016, et 83,9 %, correspondant à la résiliation à la fin de février,
      soit 59 jours sur une année de 365 jours). Appelez cette nouvelle variable "basecase_revenue_after_churn", 
      c'est-à-dire "basecase_revenue_after_churn = basecase_revenue * (1 - 0,919 * churn)".""")

    st.dataframe(df_revenu)
    st.subheader("Calculons les avantages et les coûts estimés de l'intervention")
    st.markdown(""" Maintenant, choisissez une probabilité de seuil (par exemple, 0,5) de manière à ce que :
    - Les clients avec une probabilité de résiliation plus élevée que le seuil obtiennent une remise, et
    - Les clients en dessous de la probabilité de résiliation ne bénéficient pas d'une remise.
    
    À partir de cela, calculez les revenus du scénario d'intervention en supposant que :
    - Tous les clients à qui une remise est proposée l'acceptent.
    - Les clients qui reçoivent une remise sont censés ne pas résilier au cours des douze prochains mois (c'est-à-dire
     une probabilité de résiliation de 0), et donc le revenu retenu est de 0,8 * basecase_revenue, 
     soit (1 - fraction_de_remise) * basecase_revenue.
    - Les clients qui ne reçoivent pas de remise sont supposés résilier en fonction de la variable dépendante observée
     (c'est-à-dire un 1 ou un 0 pour savoir s'ils ont effectivement résilié ou non).
     Maintenant, tracez la variation des revenus en fonction de la probabilité de seuil sur un graphique. Quelle 
     probabilité  de seuil optimise approximativement le résultat financier ? Supposons, pour ces calculs, que le 
     client ne consomme pas plus ou moins d'électricité en raison des changements de prix. En pratique, nous nous 
     attendrions à ce que si le coût du client diminue, sa consommation puisse augmenter. Nous verrons deux effets
      compensatoires en jeu :
      - Pour les vrais positifs, nous verrons la rétention de revenus par rapport au scénario sans remise.
      - Pour les faux positifs, nous verrons une réduction des revenus en leur accordant une remise alors qu'ils 
      ne résilieraient pas en réalité.
      (Les faux négatifs représentent un coût d'opportunité mais pas une différence de coût réel entre les deux
       scénarios.)
       Le point de coupure optimal équilibrera les avantages des vrais positifs par rapport aux coûts des faux 
       positifs. Notre tâche est de trouver approximativement le point de coupure optimal. Nous pourrions avoir besoin 
       de faire des hypothèses supplémentaires. Si nous pensons que les hypothèses ci-dessus ne sont pas justifiées 
       et que d'autres sont meilleures, nous devrions modifier nos hypothèses.""")
    plot_tradeoff(rev_deltas)

    st.subheader("comment choisir le seuil de coupure ?")
    st.markdown(""" Ci-dessus, nous avons décidé à qui offrir la réduction en fonction d'un seuil de probabilité. 
    Est-ce la stratégie optimale ? 
    - Par exemple, nous pourrions offrir des réductions à des clients peu rentables, 
    aggravant ainsi considérablement nos marges globales. Par exemple, si l'offre d'une réduction rend le client non 
    rentable en termes de marge nette, nous pourrions préférer les laisser partir plutôt que de les sauver.
    - Même si nous ne considérons que les revenus, cette stratégie pourrait ne pas être optimale d'un point de vue 
    financier. Nous pouvons calculer l'impact attendu sur les revenus de notre stratégie et donner la priorité aux
     clients pour lesquels la probabilité de churn est élevée, mais qui peuvent également être des clients précieux.
     
     Un principe général ici est que nous pouvons nous permettre de dépenser davantage pour conserver les clients à 
     forte valeur, car les coûts de les perdre sont plus élevés. Une erreur très courante dans les applications
      commerciales de la prévention du churn est de se concentrer sur la probabilité de churn en oubliant l'impact sur 
      la valeur (dans une plus ou moins grande mesure). Nous avons constaté de nombreux cas où nos clients consacrent 
      autant d'efforts à la fidélisation de clients peu rentables qu'à la fidélisation de clients très rentables. """)

    # Generate a list of possible cutoffs and the corresponding overall revenue deltas
    plot_tradeoff(rev_deltas_high_value)

    st.markdown(""" 
    **Note:**
    In this case, it doesn't make sense to prioritize large-revenue customers, since the overall revenue delta is much 
    lower than when targeting everyone. However, this is only the case here since the intervention doesn't depend on 
    the number of customers (simply adjusting prices). The interventions usually go beyond simply adjusting prices to 
    prevent churn.
    There may be the option of intensifying the customer relation, adding key account managers, or other interventions 
    that do incur costs depending on how many customers are targeted. In that case, it may be benefitial to target only 
    a subset of customers to save on these costs, even if the delta in the figure above is reduced.
    """)
    st.subheader(" Utilisons les prévisions plutôt que les churns réels")

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true=test_df['churn'],
        y_prob=test_df['churn_probability'],
        n_bins=10
    )
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="Our RF classifier")
    ax2.hist(test_df['churn_probability'], range=(0, 1), bins=10, histtype="step", lw=2)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots (reliability curve)')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(""" Cette représentation graphique nous indique quelques éléments :
    1. La courbe de calibration supérieure présente une courbe sigmoïde, ce qui est typique pour un classifieur 
    sous-confiant.
    2. Le graphique inférieur nous montre que le modèle est positivement biaisé vers la prédiction d'une probabilité,
     peut-être en raison d'une confiance très faible.
    En pratique, quelques ajustements seraient nécessaires pour calibrer les probabilités, mais à des fins de 
    démonstration, nous allons passer outre à cette étape.""")
    plot_tradeoff(rev_deltas)

    st.subheader("Comment choisir la remise ?")
    st.markdown(""" Dans la stratégie suggérée par le responsable de la division PME, nous offrons une remise de 20 % à tous les clients ciblés. Cependant, cela pourrait ne pas être optimal non plus. Nous avons supposé auparavant que les clients à qui une remise est offerte ne résilieront pas. Cependant, cela pourrait ne pas être vrai en réalité. La remise pourrait ne pas être suffisamment importante pour éviter la résiliation.

En fait, nous pouvons prédire la probabilité de résiliation pour chaque client en fonction du prix, de la marge et d'autres facteurs. Par conséquent, nous pouvons essayer de trouver une stratégie pour chaque client qui optimise soit leur chiffre d'affaires attendu, soit leur profit.

Pour aller plus loin, nous devrons essayer de :

- Modifier le niveau de remise offert globalement
- Prédire la réponse des clients à cette remise (c'est-à-dire la probabilité de résiliation) en fonction de la manière dont cette remise affecte leurs prix, le chiffre d'affaires et la marge.
- Faites attention à ce que nous ayons appliqué la remise à toutes les variables concernées. Pour faciliter cela, nous pourrions vouloir réentraîner notre modèle en utilisant un ensemble de variables plus simple où nous savons que nous pouvons intégrer correctement la remise dans les prédicteurs.
- Trouver le niveau de remise qui équilibre la rétention des clients par rapport au coût des faux positifs.

En fait, cela pourrait être transformé en un problème d'optimisation 2D :

- Objectif : maximiser le chiffre d'affaires net (c'est-à-dire en incluant les avantages des vrais positifs et le coût des faux positifs)
- Variables de décision :
  - Niveau de remise offert, et
  - Fraction de personnes à qui une remise est offerte

Une stratégie encore plus sophistiquée consiste à trouver le bon niveau de remise pour chaque client qui maximise
 leur chiffre d'affaires ou leur marge prévue .""")


if selected_option == "Conclusion et Recommandations" :
    st.subheader(" Le taux de résiliation est en effet élevé dans la division PME, soit 9,7 % sur 14 606 clients.")

    st.subheader("Le modèle prédictif est capable de prédire la résiliation, mais le principal facteur n'est pas"
                 "la sensibilité au prix du client. La consommation annuelle, la consommation prévue et la marge nette"
                 " sont les trois principaux facteurs.")

    st.subheader("La stratégie de réduction de 20 % est efficace, mais seulement si elle est ciblée de manière "
                 "appropriée. Offrir une réduction uniquement aux clients de grande valeur ayant une probabilité élevée "
                 "de résiliation.")



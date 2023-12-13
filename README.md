#  Projet d’analyse prédictive de la rétention client et  de la stratégie de remise de 20% ( simulation projet ) <!-- omit from toc -->

- [:mag\_right: Contexte](#mag_right-contexte)
- [🤝 Besoin Client](#besoin-client)
- [📑 Analyse du besoin client et recommandations](#analyse-besoin-client)
- [📊 Les Données](#les-donnees)
- [🗺️ Plan du Projet Data](#plan-du-projet-data)
- [📈 Résultats](#resultats)

## :mag_right: Contexte
<a name="mag_right-contexte"></a>
PowerCo, un fournisseur majeur de gaz et d'électricité, cherche à comprendre les raisons derrière le taux significatif de désabonnement  de ses clients, principalement les PME. L'entreprise explore l'hypothèse selon laquelle les variations de prix ont un impact direct sur la perte de clients.

<a name="besoin-client"></a>
## 🤝 Besoin Client
Ce projet s'inscrit dans la volonté de PowerCo d'optimiser sa stratégie de rétention des clients PME et de renforcer sa position sur le marché de l'énergie.

Plus spécifiquement, elle veut  comprendre :
Hypothèse 1 :   la sensibilité au prix est le principal facteur de churn.
Hypothèse 2 : Une remise de 20% peut empêcher un client à fort risque de résiliation contrat de partir.

<a name="analyse-besoin-client"></a>
## 📑 Analyse du besoin client et recommandations

Afin d’aider le client à optimiser  sa stratégie de rétention client, nous allons analyser les  deux hypothèses ci-dessus et apporter des réponses. 

⇒  Pour tester l'hypothèse selon laquelle le churn est influencé par la sensibilité au prix des clients, nous devons modéliser les probabilités de churn des clients et déduire l'effet des prix sur les taux de churn. Nous aurions besoin des données suivantes pour pouvoir construire les modèles.

Données nécessaires :

1. Données des clients - qui devraient inclure les caractéristiques de chaque client, par exemple, l'industrie, la consommation d'électricité historique, la date d'adhésion en tant que client, etc.
2. Données de churn - qui devraient indiquer si le client a résilié son contrat.
3. Données historiques des prix - qui devraient indiquer les prix que le client facture à chaque client pour l'électricité et le gaz à des intervalles de temps granulaires.

Une fois que nous aurons les données, le plan de travail serait le suivant :

1. Nous devrions définir ce qu'est la sensibilité au prix et la calculer.
2. Nous devrions créer des caractéristiques basées sur les données que nous obtenons, et construire un modèle de classification binaire (par exemple, Régression Logistique, Forêt Aléatoire, XGBoosting, … ).
3. Le meilleur modèle serait choisi en fonction du compromis entre la complexité, l'applicabilité et la précision des modèles.
4. Nous examinerons ensuite de manière approfondie pourquoi et comment les changements de prix impactent le churn.
5. Enfin, le modèle nous permettrait d'évaluer l'impact commercial de la stratégie de remise proposée par le client.

⇒  Pour tester l’hypothèse  selon laquelle une remise de 20% peut empêcher un client à fort risque de résiliation contrat de partir, nous allons, comme mentionné précédemment, évaluer l’impact commercial de la stratégie de remise proposée par le client.
 

<a name="les-donnees"></a>
## 📊 Les Données

PowerCo a mis à disposition un ensemble de données :

données clients qui comprennent les caractéristiques des clients PME en janvier 2016 ainsi que des informations sur s'ils ont résilié ou non leur contrat d'ici mars 2016.
Les données de prix  comprennent les tarifs de 2015 pour ces clients.

La description des différentes variables des deux bases de données se trouvent dans ce document ([data](/doc/client_and_price_data.pdf))


<a name="plan-du-projet-data"></a>
## 🗺️ Plan du Projet Data

1- Préparation des données

Il est question pour nous de nettoyer et fusionner les différentes tables. Dans le processus de nettoyages nous effectuons à la fois  des statistiques descriptives (Moyenne, médiane, écart-type)  et des graphes de visualisations ( histogramme, barplot, boxplot)  pour comprendre de manière générale chaque variable et détecter des anomalies ( outliers, asymmetries des données,  types de variables ) afin de mieux préparer les deux prochaines parties que sont le feature engineering et la modélisation. 

La préparation des données se trouve ([preprocessing](preprocessing.ipynb)). 

2- Feature engineering

Cette étape consiste à la création et/ou transformations des variables pour  préparer la phase de modélisation ([Feature engineering](feature_engineering.ipynb)). 

⇒ Nous avons  par exemple les variables liées à la consommation qui présentent une forte  asymétrie positive dont nous avons effectué des transformations logarithmiques notamment pour corriger ces asymétries.

⇒ Nous avons aussi créé des variables pour mesurer la sensibilité au prix.
- **Moyenne de la sensibilité au prix sur les 6 derniers mois, 3 derniers mois, un an :** Suivant la logique que la sensibilité au prix peut évoluer au fil du temps,  nous avons créé des caractéristiques qui capturent les variations récentes et peuvent révéler des changements de comportement plus rapides. En utilisant une fenêtre d’un an, de  6 mois, de 3 mois,  nous avons créé de nouvelles caractéristiques pour mesurer la sensibilité au prix en utilisant la moyenne de l'année, des 6 derniers mois et des 3 derniers mois.
- **Variation moyenne des prix entre les heures de pointe et les heures creuses :** Les clients peuvent être sensibles aux variations de prix en fonction de la période de la journée. Cette caractéristique pourrait refléter la réaction des clients aux fluctuations de prix pendant les heures de pointe par rapport aux heures creuses
- **Différence entre les prix hors pointe en décembre et janvier précédent :** Les variations de prix d'un mois à l'autre, en particulier pendant la saison des fêtes, peuvent influencer la sensibilité au prix
- **Variation moyenne des prix sur les périodes :** cette caractéristique examine la variation moyenne des prix sur différentes périodes (hors pointe, pointe, mi-pointe).
- **Changements de prix maximaux à travers les périodes et les mois :** Cette caractéristique explore les variations maximales de prix, ce qui peut être perçu comme plus impactant pour les clients. Des changements brusques et importants peuvent influencer la décision de résilier un contrat.

  **Note:**
Nous aurions pu continuer par exemple en créant des variables sur la comparaison des prix actuels avec les prévisions, Fréquence des changements de prix, Comparaison des tarifs avec la concurrence. 

Et aussi réfléchir à l'analyse des séries temporelles pourrait apporter une perspective supplémentaire à votre étude, notamment en tenant compte de l'évolution des prix au fil du temps

Une autre chose utile à examiner est la corrélation entre toutes les variables de votre ensemble de données.

3- Construire les modèles de machine learning : 
 
Nous nous concentrons uniquement sur l'entraînement d'un Random Forest ([Feature engineering.ipynb](modelling_model.ipynb)) 

4- Evaluation du modèles et interprétations

 Métriques de mesure de performance :
- Accuracy = le rapport des observations correctement prédites sur l'ensemble des observations
- Précision = la capacité du classificateur à ne pas étiqueter un échantillon négatif comme positif
- Rappel = la capacité du classificateur à trouver tous les échantillons positifs

5-   Analyse de l’impact Business de la Remise de 20% ([Impact_Business_Remise.ipynb](discount_impact.ipynb))

Nous effectuons une analyse de la stratégie de remise proposée. L'entreprise a proposé d'offrir une remise de 20% aux clients ayant une forte propension à la résiliation.

Nous pouvons supposer, pour commencer, que tous ceux à qui une remise est offerte l'accepteront

Notre tâche est de calculer le chiffre d'affaires prévu pour l'ensemble des clients :
- Lorsqu'aucune remise n'est proposée
- Lorsqu'une remise est offerte en fonction d'un seuil de probabilité pour décider qui devrait recevoir la remise de 20%
  
Ainsi, décider où fixer le seuil pour maximiser le chiffre d'affaires.


<a name="resultats"></a>
## 📈 Résultats

-  **📈 Le taux de résiliation est en effet élevé dans la division des PME**

  • 9,7 % sur 14 606 clients

- **Le modèle prédictif est capable de prédire le désabonnement, mais le principal facteur n'est pas la sensibilité au prix du client**

• La consommation annuelle, la consommation prévue et la marge nette sont les trois principaux moteurs.

- ** 🔖 Stratégie de remise de 20% est efficace mais seulement si elle est ciblée de manière appropriée**

• Offrir une remise uniquement aux clients de grande valeur avec une probabilité de désabonnement élevée




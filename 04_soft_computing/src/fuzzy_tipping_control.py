import numpy as np 
import skfuzzy as fuzz
from skfuzzy import control as ctrl 


#1- Define variables : input and output 
#variables floues d entrees : température et humidité
#variables floues de sortie : vitesse du ventilateur


# crée une plage de valeurs de température allant de 15 à 40 (inclus) avec un pas de 1
temperature = ctrl.Antecedent(np.arange(15, 41, 1), 'temperature') 
#np.arange(0, 101, 1) crée une plage de valeurs d'humidité allant de 0 à 100 (inclus) avec un pas de 1. 3. 
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
#conséquent" qui est la variable de sortie du système de logique floue
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

#2- Fonctions d appartenance: définissent le degré d'appartenance d'une valeur spécifique

# For temperature: utilise la biblio ppiur cree fct d appartence de chauqe variable 
temperature['low'] = fuzz.trimf(temperature.universe, [15, 15, 25])
temperature['medium'] = fuzz.trimf(temperature.universe, [20, 25, 30])
temperature['high'] = fuzz.trimf(temperature.universe, [25, 40, 40])

# For humidity
humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['medium'] = fuzz.trimf(humidity.universe, [25, 50, 75])
humidity['high'] = fuzz.trimf(humidity.universe, [50, 100, 100])

# For the speed of the fan
fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [25, 50, 75])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [50, 100, 100])

#La fonction fuzz.trimf est utilisée pour créer des fonctions d'appartenance triangulaires




#3- definir les regles floues : 

#Règle 1: Si la température et l'humidité sont élevées, la vitesse du ventilateur doit l'être également.
rule1 = ctrl.Rule(temperature['high'] & humidity['high'], fan_speed['high'])
rule2 = ctrl.Rule(temperature['medium'] & humidity['high'], fan_speed['medium'])
rule3 = ctrl.Rule(temperature['low'] & humidity['low'], fan_speed['low'])



#4 - Systemes de Control : compile les règles floues dans un modèle de travail

fan_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
fan_simulation = ctrl.ControlSystemSimulation(fan_ctrl)



#5- Saisir les valeurs et effecture les calculs 
#Le système traite ces données à l'aide de règles floues et calcule la vitesse du ventilateur correspondant.

# Input values
fan_simulation.input['temperature'] = 32  # Example temperature (°C)
fan_simulation.input['humidity'] = 75    # Example humidity (%)

# Perform computation
fan_simulation.compute()

#Le système produit une valeur brute pour la vitesse du ventilateur sur la base du calcul de la logique floue


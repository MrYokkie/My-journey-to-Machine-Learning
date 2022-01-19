import numpy as np # import numpy dla raboty
def sigmoid(x):  #sozdanie funkcji sigmoid w naszem sluczaje funkcjia aktiwator
	return 1/(1 + np.exp(-x))
#opis wwodnych dannych
training_inputs= np.array([[0,0,1],
						  [1,1,1],
						  [1,0,1],
						  [0,1,1]]) 	#objawlenie trenirowocznych dannych- sozdanie masiwa s wchodnymi dannymi

training_outputs= np.array([[0,1,1,0,1,0,1]]).T	#wwod ozydajemych dannych-takze transponirowanie 4 na 1
#dalee opisanie wesow to est synapsow
np.random.seed(1) #ispolzujetsa generator random czisel, jedenica dla poluczenia takogoze czisla kak u haudy

synaptic_wieghts = 2* np.random.random((3,1)) - 1 #inicializacja wesow po puti sozdania masiwa 3 na 1

print("Random weights inicialization:")
print(synaptic_wieghts)
#Metod obratnogo rasprostaranenia
for i in range(500):
	input_layer = training_inputs
	outputs = sigmoid( np.dot(input_layer, synaptic_wieghts))
	err = training_outputs - outputs
	adjustments = np.dot(input_layer.T, err *(outputs*(1 - outputs)))
	synaptic_wieghts += adjustments

print("Wesa posle obuczenia:")
print(synaptic_wieghts)

print("Results after studying")
print(outputs)
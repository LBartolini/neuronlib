import neuronlib as nl
import matplotlib.pyplot as plt

X = [  # milion of people, 1000 km surface area, hundreds of municipality | 0= big city(chief town), 1= little city
    [0.49, 7.7, 0.92, 1, 'Sassari'],
    [0.52, 7.4, 1.1, 1, 'Bolzano'], # bolzano
    [0.62, 7.0, 0.61, 1, 'Foggia'], # foggia
    [0.58, 6.8, 2.5, 1, 'Cuneo'], # cuneo
    [2.2, 6.8, 3.1, 0, 'Torino'], # torino
    [0.70, 6.7, 1.5, 1, 'Cosenza'], # cosenza
    [0.36, 6.5, 1.0, 1, 'Potenza'], # potenza
    [0.65, 6.3, 0.59, 1, 'Perugia'], # perugia
    [0.53, 6.2, 1.76, 1, 'Trento'], # trento
    [0.21, 5.6, 0.74, 1, 'Nuoro'], # Nuoro
    [4.3, 5.3, 1.2, 0, 'Roma'], # roma
    [0.30, 5.0, 1.1, 1, 'Aquila'], # l'aquila
    [1.23, 5.0, 0.8, 0, 'Palermo'], # palermo
    [0.53, 4.9, 1.34, 1, 'Udine'], # udine
    [1.1, 4.9, 1.58, 1, 'Salerno'], # salerno
    [1.2, 4.7, 2.0, 1, 'Brescia'], # brescia
    [0.22, 4.5, 0.28, 1, 'Grosseto'], # grosseto
    [1.2, 3.8, 0.41, 0, 'Bari'], # bari
    [0.26, 3.8, 0.35, 1, 'Siena'], # siena
    [1.1, 3.7, 0.55, 0, 'Bologna'], # bologna
    [0.31, 3.6, 0.6, 1, 'Viterbo'], # viterbo
    [0.20, 3.6, 0.63, 1, 'Belluno'], # belluno
    [1.1, 3.5, 0.58, 0, 'Catania'], # catania
    [0.42, 3.5, 1.88, 1, 'Alessandria'], # alessandria
    [1.0, 3.5, 0.42, 0, 'Firenze'], # firenze
    [0.19, 3.4, 0.31, 1, 'Matera'], # matera
    [0.45, 3.4, 0.45, 1, 'Parma'], # parma
    [0.63, 3.2, 1.08, 0, 'Messina'], # messina
    [0.12, 3.2, 0.74, 1, 'Aosta'], # aosta
    [0.49, 3.2, 0.91, 1, 'Frosinone'] # frosinone
    ]

Ag = nl.Neurone(3, lr=0.005) # n of inputs and learning rate (dafault is 0.01)

if True:
    for epoch in range(5001):
        if epoch % 1000 == 0 and epoch > 0:
            for i, x in enumerate(X):
                target = x[3]
                input_nn = [x[0], x[1], x[2]]
                print(input_nn, target)
                pred = Ag.learn(input_nn, target, to_print=True)
                if pred >= 0.75:
                    plt.plot(x[0], x[1], 'ro', color='red', label=x[4])
                elif pred <= 0.25:
                    plt.plot(x[0], x[1], 'ro', color='blue', label=x[4])
                else:
                    plt.plot(x[0], x[1], 'ro', color='black', label=x[4])
            plt.xlabel('Popolazione')
            plt.ylabel('Superficie')
            plt.legend()
            print('\n\n%s\n\n'%epoch)
            plt.show()
        else:
            for i, x in enumerate(X):
                target = x[3]
                Ag.learn([x[0], x[1], x[2]], target, to_print=True)

    #print(Ag.weights, Ag.bias)

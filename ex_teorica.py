"""
Rede Bayesiana presente na aula teórica disponibilizado pelo Professor Doutor Luís Alexandre, Departamento de Informática
da Universidade da Beira Interior
Feito por João Miguel Santos a50367 - LEI

A rede bayesiana vai estar presente no repo em https://github.com/jmiguelsantos2047


Crie um modelo desta rede Bayesiana usando um código semelhante ao desenvolvido
nos exercícios anteriores (veja o ficheiro ex3.py), notando que
agora as variáveis em vez de tomarem valores ’A’, ’B’ e ’C’, tomam valores
’V’ ou ’F’ apenas.

Biblioteca(s) usadas: pyAgrum <3
"""


import pyAgrum as agru

def mostra(d,q):
    global ie
    ie.setEvidence(d)
    ie.makeInference()
    print(ie.posterior(q))

bn = agru.BayesNet('Assaltos na Vivenda')

assalto = bn.add(agru.LabelizedVariable('assalto','assalto',2))
tremor = bn.add(agru.LabelizedVariable('tremor','tremor',2))
alarme = bn.add(agru.LabelizedVariable('alarme','alarme', 2))
joaoliga = bn.add(agru.LabelizedVariable('joaoliga', 'joaoliga', 2))
marialiga = bn.add(agru.LabelizedVariable('marialiga', 'marialiga', 2))


bn.addArc(assalto,alarme)
bn.addArc(tremor,alarme)
bn.addArc(alarme, joaoliga)
bn.addArc(alarme, marialiga)

bn.cpt(assalto)[{}] = [0.999, 0.001]
bn.cpt(tremor)[{}] = [0.998, 0.002]

bn.cpt(alarme)[{'tremor': 1, 'assalto': 1}] = [0.05,0.95]
bn.cpt(alarme)[{'tremor': 1, 'assalto': 0}] = [0.71, 0.29]
bn.cpt(alarme)[{'tremor': 0, 'assalto': 1}] = [0.06, 0.94]
bn.cpt(alarme)[{'tremor': 0, 'assalto': 0}] = [0.99, 0.01]

bn.cpt(joaoliga)[{'alarme': 1}] = [0.10,0.90]
bn.cpt(joaoliga)[{'alarme': 0}] = [0.95, 0.05]

bn.cpt(marialiga)[{'alarme': 1}] = [0.30,0.70]
bn.cpt(marialiga)[{'alarme': 0}] = [0.99, 0.01]

ie = agru.LazyPropagation(bn)

# probabilidade da maria ligar?
mostra({'alarme': 1}, 'marialiga')
print("Probabilidade da Maria ligar sendo que o alarme disparou? P(M|A)?", ie.posterior('marialiga')[1])

# probabilidade do joao ligar sendo que o alarme nao disparou
mostra({'alarme': 0}, 'joaoliga')
print("Probabilidade do João ligar sendo que o alarme não disparou? P(J=1|A=0):  ", ie.posterior('joaoliga')[0])

# probabilidade do alarme disparar dado que nao houve tremor de terra nem assalto
mostra({'tremor':0}, 'alarme')
print("Probabilidade do alarme disparar sendo que não houve tremor de terra? P(A=1|T=0)", ie.posterior('alarme')[1])

#probabilidade alarme dispara dado que houve tremor de terra e assalto
mostra({'tremor':1,'assalto':1}, 'alarme')
print("Probabilidade do alarme disparar sendo que houve tremor de terra e assalto? P(A=1|T=1,S=1)  ", ie.posterior('alarme')[1])


#podem continuar a calcular as probabilidades usando a função mostrar!!! have fun!
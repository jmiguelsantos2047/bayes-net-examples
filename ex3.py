"""

Resolução do exercício 3 da ficha 5 disponibilizado pelo Professor Doutor Luís Alexandre, Departamento de Informática
da Universidade da Beira Interior
Feito por João Miguel Santos a50367 - LEI


Crie um modelo desta rede Bayesiana usando um código semelhante ao desenvolvido
nos exercícios anteriores (veja o ficheiro ex3.py), notando que
agora as variáveis em vez de tomarem valores ’A’, ’B’ e ’C’, tomam valores
’V’ ou ’F’ apenas. Agora use o seu modelo para responder às questões presentes
no final do ficheiro.

Biblioteca(s) usadas: pyAgrum <3




"""



import pyAgrum as agru

def mostra(d, q):
    # mostra resultado do query q, dadas as observações d
    global ie
    ie.setEvidence(d)
    ie.makeInference()
    print (ie.posterior(q))

bn=agru.BayesNet('RelvaMolhada')

# criar nodos
chuva=bn.add(agru.LabelizedVariable('chuva','chuva',2))
aspersor=bn.add(agru.LabelizedVariable('aspersor','aspersor ligado',2))
relva=bn.add(agru.LabelizedVariable('relva','relva molhada',2))

# criar arestas
bn.addArc(chuva,aspersor)
bn.addArc(chuva,relva)
bn.addArc(aspersor, relva)
# COMPLETAR

# colocar tabelas de probabilidade nos nodos
bn.cpt(chuva)[{}] = [0.8,0.2] # P(Chuva=0) e P(Chuva=1)

bn.cpt(aspersor)[{'chuva': 0}] = [0.6, 0.4] # P(Aspersor=0|Chuva=0), P(Aspersor=1|Chuva=0)
bn.cpt(aspersor)[{'chuva': 1}] = [0.99, 0.01] # P(Aspersor=0|Chuva=1), P(Aspersor=1|Chuva=1)

bn.cpt(relva)[{'chuva': 0, 'aspersor': 0}] = [1, 0]
bn.cpt(relva)[{'chuva': 0, 'aspersor': 1}] = [0.1, 0.9]
bn.cpt(relva)[{'chuva': 1, 'aspersor': 0}] = [0.2, 0.8]
bn.cpt(relva)[{'chuva': 0, 'aspersor': 1}] = [0.01, 0.99]
# COMPLETAR

ie=agru.LazyPropagation(bn)

# 3a: Qual é a probabilidade de a relva não estar molhada?
# COMPLETAR
mostra({}, 'relva') # mostra prob de cada estado da variavel relva
print("Relva = 0 (não está molhada):  ", ie.posterior('relva')[0])

# 3b: Qual é a probabilidade de estar a chover dado que observamos a relva molhada?
# COMPLETAR
mostra({'relva': 1}, 'chuva') # evidencia que a relva esta molhada(relva=1)
print("Probabilidade de estar a chover dado que a relva está molhada (relva=1)", ie.posterior('chuva')[1])


# 3c: Qual é a probabilidade de o aspersor estar desligado dado que a relva não está molhada e não está a chover?
# COMPLETAR
mostra({'relva': 0, 'chuva': 0}, 'aspersor') # evidencia que a relva nao esta molhada(relva=0) e nao esta a chover(chuva=0)
print("Probabilidade de o aspersor estar desligado (Aspersor=0) dado que a relva nao esta molhada e nao esta a chover: ", ie.posterior('aspersor')[0])

import random, math

def computeSinDist(x):
    return (math.sin(x * math.pi))

def computeAmountToSpend(budget, l):
    output = []
    total = 0

    # find total sum
    for user in l:
        total += computeSinDist(user)

    # compute each allocation amount
    for user in l:
        output.append((user, (computeSinDist(user)/total) * budget))

    return output

def prettyOutput(amnts):
    for amn in amnts:
        print(f'Likelihood: {amn[0]} | Allocation: {amn[1]}')

def genProbs(numOfPeople):
    l = []

    for x in range(numOfPeople):
        l.append(random.random())

    return l

def computeUniSum(probs, budgetPerPerson):
    uniSum = 0

    for i in range(0, len(probs)):
        uniSum += budgetPerPerson * probs[i]

    return uniSum

def computeSmartSum(allocs):
    smartSum = 0

    for prob, alloc in allocs:
        smartSum += alloc * prob

    return smartSum
    

def main():
    numOfPeople = 1000
    budget = 10000
    # allocation for uniform distribution
    budgetPerPerson = budget / numOfPeople
    # probability-based allocation
    probs = genProbs(numOfPeople)
    allocs = computeAmountToSpend
    (budget, probs)
    # determine effectivness of 
    # probability-based allocation
    uniSum = computeUniSum(probs,budgetPerPerson)
    smartSum = computeSmartSum(allocs)
    print(f'Uniform: {uniSum}')
    print(f'Probability Based: {smartSum}')
main()
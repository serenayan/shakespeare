import random
import operator
import numpy as np

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = np.array(A)
        self.O = np.array(O)
        self.A_start = [1. / self.L for _ in range(self.L)]
    
    def getA():
        for a in self.A:
            print(a)
    def getO():
        for o in self.O:
            print(o)

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]
        t = []
        # initialize row 2 of probs and seqs as 0
        for i in range(self.L):
            probs[1][i] = self.A_start[i] * self.O[i][x[0]]

        # start w row 2 and use top-down approach
        for row in range(2, M+1):
            for col1 in range(self.L):
                temp = []
                for col2 in range(self.L):
                    temp.append(probs[row-1][col2] * self.O[col1][x[row-1]] *
                                self.A[col2] [col1])
                max_index, max_value = max(enumerate(temp), key=operator.itemgetter(1))
                probs[row][col1] = max_value
                seqs[row][col1] = seqs[row-1][max_index] + str(max_index)

        max_index, max_value = max(enumerate(probs[M]), key=operator.itemgetter(1))
        max_seq = seqs[len(probs)-1][max_index] + str(max_index)

        return max_seq


    def forward(self, x, normalize=True):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''
        M = len(x)
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(self.L):
            alphas[1][i] = self.O[i][x[0]] * self.A_start[i]

        for i in range(1, M):
            for j in range(self.L):
                sum = 0
                for l in range(self.L):
                    sum += alphas[i][l] * self.A[l][j]
                alphas[i + 1][j] = self.O[j][x[i]] * sum

        if (normalize):
            for i in range(1, M):
                sum = 0
                for a in alphas[i + 1]:
                    sum += a
                if sum != 0:
                    alphas[i + 1] = [a / sum for a in alphas[i + 1]]

        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''
        M = len(x)      # len of sequence
        if normalize:
            betas = [[0. for i in range(self.L)] for j in range(M + 1)]

            for i in range(self.L):
                betas[M][i] = 1

            for row in range(M-1, 0, -1):
                for col in range(self.L):
                    for col2 in range(self.L):
                        betas[row][col] += (betas[row+1][col2] *
                                            self.O[col2][x[row]] * self.A[col] [col2])
                norm = sum(betas[row])
                for nCol in range(self.L):
                    betas[row][nCol] /= norm
            return betas

        betas = [[0. for i in range(self.L)] for j in range(M + 1)]

        for i in range(self.L):
            betas[M][i] = 1

        for row in range(M-1,0, -1):
            for col in range(self.L):
                for col2 in range(self.L):
                    betas[row][col] += (betas[row+1][col2] * self.O[col2][x[row]] *
                                self.A[col] [col2])
        return betas

    def count_transitions(self, a, b, X, Y):
        '''
        y_i^j = b
        and y_i^(j-1) = a
        '''
        den = 0
        num = 0

        for i in range(len(X)):
            for j in range(1, len(X[i])):
                if Y[i][j-1] == a:
                    den += 1
                    if Y[i][j] == b:
                        num += 1

        return num, den

    def count_observations(self, w, a, X, Y):
        '''
        y_i^j = w
        and x_i^j = w
        '''
        den = 0
        num = 0

        for i in range(len(X)):
            for j in range(len(X[i])):
                if Y[i][j] == a:
                    den += 1
                    if X[i][j] == w:
                        num += 1
        return num, den

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        for a in range(self.L):
            for b in range(self.L):
                num, den = self.count_transitions(a, b, X, Y)
                self.A[a][b] = num / den

        for a in range(len(self.O)):
            for w in range(len(self.O[0])):
                num, den = self.count_observations(w, a, X, Y)
                self.O[a][w] = num / den

        pass


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        for x in range(N_iters):
            print("Iteration " + str(x))

            A_den = [0. for i in range(self.L)]
            O_den = [0. for i in range(self.L)]

            A_num = [[0. for i in range(self.L)] for i in range(self.L)]
            O_num = [[0. for i in range(self.D)] for i in range(self.L)]

            for sequence in X:
                M = len(sequence)

                alpha = self.forward(sequence, normalize=True)
                beta = self.backward(sequence, normalize=True)

                for w in range(1, M+1):
                    temp = [0. for i in range(self.L)]
                    for z in range(self.L):
                        temp[z] = alpha[w][z] * beta[w][z]

                    # normalize
                    norm = sum(temp)

                    for z in range(len(temp)):
                        temp[z] /= norm

                    for z in range(self.L):
                        if w != M:
                            A_den[z] += temp[z]
                        O_num[z][sequence[w-1]] += temp[z]
                        O_den[z] += temp[z]

                for w in range(1, M):
                    norm = 0

                    temp = [[0. for _ in range(self.L)] for _ in range(self.L)]

                    for i in range(self.L):
                        for j in range(self.L):
                            temp[i][j] = alpha[w][i] * self.A[i][j] * self.O[j][sequence[w]] * beta[w+1][j]

                    for row in temp:
                        norm += sum(row)

                    for i in range(self.L):
                        for j in range(self.L):
                            temp[i][j] /= norm

                    for i in range(self.L):
                        for j in range(self.L):
                            A_num[i][j] += temp[i][j]

            for i in range(self.L):
                for j in range(self.L):
                    self.A[i][j] = A_num[i][j] / A_den[i]

            for i in range(self.L):
                for j in range(self.D):
                    self.O[i][j] = O_num[i][j] / O_den[i]


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []
        state = random.choice(range(self.L))

        for _ in range(M):
            states.append(state)

            t = self.A[state]
            o = self.O[state]

            emission.append(int(np.random.choice(range(self.D), 1, p=o)))

            state = int(np.random.choice(range(self.L), 1, p=t))

        return emission, states

    def generate_emission_rhyme_meter(self, M, mapping, mapping_r, rhyme, rhyme_dic, stress_dic):
        '''
        Generates  emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Returns:
            emission:   The randomly generated emission as a list.
            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        # generate ending word
        ending_word = random.choice(rhyme_dic[rhyme])
        ending_emission = mapping[ending_word]
        emission.append(ending_emission)

        # reverse generate state of the words
        prob_state = self.O[:, ending_emission]
        prob_state /= np.sum(prob_state)
        state = int(np.random.choice(range(self.L), 1, p=prob_state))
        states.append(state)

        stress_pattern = stress_dic[ending_word] # must keep in normal order
        count = M - len(stress_pattern)
        while count > 0:
            # last state
            state = states[-1]
            t = self.A[:, state]
            t/= np.sum(t)
            state = int(np.random.choice(range(self.L), 1, p=t))
            
            # last emission for the state
            o = self.O[state]
            o/= np.sum(o)
          
            next_obs = int(np.random.choice(range(self.D), 1, p=o))
            # check emission stress pattern
            if mapping_r[next_obs] in stress_dic:
                curr_stress = stress_dic[mapping_r[next_obs]]
                if (curr_stress[-1] != stress_pattern[0]) and (count - len(curr_stress) >= 0):
                    emission.append(next_obs)
                    stress_pattern = curr_stress + stress_pattern
                    count -= len(curr_stress)
                    states.append(state)
                        


        return reversed(emission), reversed(states)


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM

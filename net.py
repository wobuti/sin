import chainer
import chainer.functions as F
import chainer.links as L
class RNNLM(chainer.Chain):

    """Recurrent neural net languabe model for penn tree bank corpus.

    This is an example of deep LSTM network for infinite length input.

    """
    def __init__(self, length_of_sequences, n_units, train=True):
        super(RNNLM, self).__init__(
            l0=L.Linear(length_of_sequences, 1),
            l1=L.LSTM(1, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, 1),
        )
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x,t):
        y = self.predict(x)
        self.loss = F.mean_squared_error(y, t)
        return self.loss

    def predict(self, x):
        # h0 = self.l0(F.dropout(x_data, train=self.train))
        # h1 = self.l1(F.dropout(h0, train=self.train))
        # h2 = self.l2(F.dropout(h1, train=self.train))
        # h3 = self.l3(F.dropout(h2, train=self.train))
        h0 = self.l0(x)
        h1 = self.l1(h0)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        return h3




import numpy as np
import sklearn
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge

def graipher(pts, K, start=False):
    farthest_pts = np.zeros((K, pts.shape[1]))
    farthest_pts_ids = []
    if start:
        farthest_pts[0] = start
    else:
        farthest_pts[0] = pts[np.random.randint(len(pts))]

    farthest_pts_ids.append(np.random.randint(len(pts)))
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        farthest_pts_ids.append(np.argmax(distances))

        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts, farthest_pts_ids


def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


class KernelFit:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.alpha = None
        self.kernel = None
        self.models = []
        self.scale_parms = []
        self.r2s = []
        self.test_results = []
        self.lcs = None
        self.test_ids = None
        self.train_ids = None

    def __int__(self):
        self.init()

    def init(self):
        self.models = []
        self.scale_parms = []
        self.r2s = []
        self.test_results = []
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.alpha = None
        self.kernel = None
        self.test_ids = None
        self.train_ids = None

    def set_data(self, distM, ids, lcs):
        self.X = distM
        self.y = lcs
        self.ids = ids

    def fit(self,
            alpha=1e-3,
            N_SAMPLE_POINTS=None,
            start=False,
            model_type=KernelRidge,
            kernel=RBF()):
        """

        :param alpha:
        :param N_SAMPLE_POINTS:
        :param start:
        :return:
        """

        if N_SAMPLE_POINTS is None:
            N_SAMPLE_POINTS = len(self.X)//10
            print("N_SAMPLE_POINTS set to {}".format(N_SAMPLE_POINTS))

        points, ids = graipher(
            self.X,
            N_SAMPLE_POINTS,
            start=start
        )
        npoints = len(self.X)
        inx_vals = np.arange(npoints)
        self.train_ids = ids
        test_ids = np.delete(inx_vals, ids, axis=0)
        self.test_ids = test_ids
        self.X_train = [self.X[i] for i in ids]
        self.X_test = [self.X[i] for i in test_ids]

        # a kernel for each axis of each charge
        for chgindx in range(self.y.shape[1]):
            lcs_ = np.array([np.array(_).flatten()[chgindx]
                             for _ in self.y])
            y = lcs_
            y_train = np.array([y[i] for i in ids])
            y_test = np.array([y[i] for i in test_ids])

            model = model_type(
                alpha=alpha,
                kernel=kernel,
            ).fit(self.X_train, y_train)

            # evaluate the model
            train_predictions = model.predict(self.X_train)
            test_predictions = model.predict(self.X_test)

            r2_train = sklearn.metrics.r2_score(y_train,
                                                train_predictions)
            r2_test = sklearn.metrics.r2_score(y_test,
                                               test_predictions)
            #  save the model
            self.models.append(model)
            self.scale_parms.append((lcs_.min(), lcs_.max()))
            self.r2s.append([r2_test, r2_train])
            self.test_results.append((y_test, test_predictions))

    def predict(self, X):
        return np.array([model.predict(X)
                         for model in self.models])

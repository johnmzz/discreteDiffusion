import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import math
from torch_scatter import scatter
from torch_geometric.utils import to_dense_batch

class AttentionModule(torch.nn.Module):
    def __init__(self, para_dim):
        super(AttentionModule, self).__init__()
        self.para_dim = para_dim
        self.setup_weights()
        self.init_parameters()

    # define weights
    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.para_dim, self.para_dim))

    # init the attention parameters
    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, batch, size=None):
        """"
         Making a forward propagation pass to create a graph level representation.
        :param embedding: Graph level embedding of the  GCN.
        :return representation: A graph level representation vector.
        """
        # global_context = torch.mean(torch.matmul(embedding, self.weight_matrix),dim=0) # 支持多维的矩阵乘法 注意这里dim=0表示按行求平均
        # transform_global = torch.tanh(global_context)
        # sigmoid_scores = torch.sigmoid(torch.mm(embedding, transform_global.view(-1, 1)))    # 单纯的矩阵乘法
        # representation = torch.mm(torch.t(embedding), sigmoid_scores)
        # return representation
        size = batch[-1].item() + 1 if size is None else size
        mean = scatter(x, batch, dim=0, dim_size=size, reduce='mean')       #scatter 分组合并。128 * 64
        transformed_global = torch.tanh(torch.mm(mean, self.weight_matrix))
        coefs = torch.sigmoid((x * transformed_global[batch] * 10).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x
        return scatter(weighted, batch, dim=0, dim_size=size, reduce='add')


class NeuralTensorNetwork(torch.nn.Module):
    """
    Common used, Tensor Network module to calculate similarity vector, 19 WSDM, 21 ICDE and etc
    """
    def __init__(self, args):
        super(NeuralTensorNetwork, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weights = torch.nn.Parameter(torch.Tensor(self.args.filter_3, self.args.filter_3, self.args.tensor_neurons))
        self.weights_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 2*self.args.filter_3))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weights)
        torch.nn.init.xavier_uniform_(self.weights_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
          Making a forward propagation pass to create a similarity vector.
          :param embedding_1: Result of the 1st embedding after attention.
          :param embedding_2: Result of the 2nd embedding after attention.
          :return scores: A similarity score vector.
        """
        scoring = torch.mm(torch.t(embedding_1), self.weights.view(self.args.filter_3, -1))
        scoring = scoring.view(self.args.filter_3, self.args.tensor_neurons)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weights_matrix_block,combined_representation)
        scores = F.relu(scoring + block_scoring + self.bias)
        return scores


class Affinity(torch.nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        self.A = torch.nn.Parameter(torch.Tensor(self.d, self.d))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.d)

    def forward(self, X, Y):
        assert X.shape[1] == Y.shape[1] == self.d
        M = torch.matmul(X, self.A)
        M = torch.matmul(M, Y.transpose(0, 1))
        return M
        # assert X.shape[2] == Y.shape[2] == self.d
        # M = torch.matmul(X, self.A)
        # M = torch.matmul(M, Y.transpose(1, 2))
        # return M

    def forward(self, feature_1, feature_2, batch_1, batch_2):
        dense_feature_1, mask_1 = to_dense_batch(feature_1, batch_1)
        dense_feature_2, mask_2 = to_dense_batch(feature_2, batch_2)
        b_mat = torch.matmul(dense_feature_1, dense_feature_2.permute([0, 2, 1]))
        # b_mat = torch.matmul(dense_feature_1, dense_feature_2.transpose(1,2))
        return b_mat

class AffinityInp(torch.nn.Module):
    """
    Affinity Layer to compute inner product affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d):
        super(AffinityInp, self).__init__()
        self.d = d

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        M = torch.matmul(X, Y.transpose(1, 2))
        return M

class NorSim(torch.nn.Module):

    def __init__(self):
        super(NorSim, self).__init__()

    def forward(self, sim_mat: Tensor, nrows: Tensor=None, ncols: Tensor=None):
        batch_size = sim_mat.shape[0]
        ret_sim = torch.full((sim_mat.shape), 0.0, device=sim_mat.device, dtype=sim_mat.dtype)

        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            b_sim_mat = sim_mat[b, row_slice, col_slice]
            b_sim_mat = torch.softmax(b_sim_mat, dim=-1)
            ret_sim[b, row_slice, col_slice] = b_sim_mat
        return ret_sim


class Sinkhorn(nn.Module):
    r"""
    Sinkhorn algorithm turns the input matrix into a bi-stochastic matrix.
    Sinkhorn algorithm firstly applies an ``exp`` function with temperature :math:`\tau`:
    .. math::
        \mathbf{S}_{i,j} = \exp \left(\frac{\mathbf{s}_{i,j}}{\tau}\right)
    And then turns the matrix into doubly-stochastic matrix by iterative row- and column-wise normalization:
    .. math::
        \mathbf{S} &= \mathbf{S} \oslash (\mathbf{1}_{n_2} \mathbf{1}_{n_2}^\top \cdot \mathbf{S}) \\
        \mathbf{S} &= \mathbf{S} \oslash (\mathbf{S} \cdot \mathbf{1}_{n_2} \mathbf{1}_{n_2}^\top)
    where :math:`\oslash` means element-wise division, :math:`\mathbf{1}_n` means a column-vector with length :math:`n`
    whose elements are all :math:`1`\ s.
    :param max_iter: maximum iterations (default: ``10``)
    :param tau: the hyper parameter :math:`\tau` controlling the temperature (default: ``1``)
    :param epsilon: a small number for numerical stability (default: ``1e-4``)
    :param log_forward: apply log-scale computation for better numerical stability (default: ``True``)
    :param batched_operation: apply batched_operation for better efficiency (but may cause issues for back-propagation,
     default: ``False``)
    .. note::
        ``tau`` is an important hyper parameter to be set for Sinkhorn algorithm. ``tau`` controls the distance between
        the predicted doubly-stochastic matrix, and the discrete permutation matrix computed by Hungarian algorithm (see
        :func:`~src.lap_solvers.hungarian.hungarian`). Given a small ``tau``, Sinkhorn performs more closely to
        Hungarian, at the cost of slower convergence speed and reduced numerical stability.
    .. note::
        We recommend setting ``log_forward=True`` because it is more numerically stable. It provides more precise
        gradient in back propagation and helps the model to converge better and faster.
    .. note::
        Setting ``batched_operation=True`` may be preferred when you are doing inference with this module and do not
        need the gradient.
    """
    def __init__(self, max_iter: int=10, tau: float=1., epsilon: float=1e-4,
                 log_forward: bool=True, batched_operation: bool=False):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.tau = tau
        self.epsilon = epsilon
        self.log_forward = log_forward
        if not log_forward:
            print('Warning: Sinkhorn algorithm without log forward is deprecated because log_forward is more stable.')
        self.batched_operation = batched_operation # batched operation may cause instability in backward computation,
                                                   # but will boost computation.

    def forward(self, s: Tensor, nrows: Tensor=None, ncols: Tensor=None, dummy_row: bool=False) -> Tensor:
        r"""
        :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
        :param nrows: :math:`(b)` number of objects in dim1
        :param ncols: :math:`(b)` number of objects in dim2
        :param dummy_row: whether to add dummy rows (rows whose elements are all 0) to pad the matrix to square matrix.
         default: ``False``
        :return: :math:`(b\times n_1 \times n_2)` the computed doubly-stochastic matrix
        .. note::
            We support batched instances with different number of nodes, therefore ``nrows`` and ``ncols`` are
            required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
            the batched matrices are not padded.
        .. note::
            The original Sinkhorn algorithm only works for square matrices. To handle cases where the graphs to be
            matched have different number of nodes, it is a common practice to add dummy rows to construct a square
            matrix. After the row and column normalizations, the padded rows are discarded.
        .. note::
            We assume row number <= column number. If not, the input matrix will be transposed.
        """
        if self.log_forward:
            return self.forward_log(s, nrows, ncols, dummy_row)
        else:
            return self.forward_ori(s, nrows, ncols, dummy_row) # deprecated

    def forward_log(self, s, nrows=None, ncols=None, dummy_row=False):
        """Compute sinkhorn with row/column normalization in the log space."""
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        if s.shape[2] >= s.shape[1]:
            transposed = False
        else:
            s = s.transpose(1, 2)
            nrows, ncols = ncols, nrows
            transposed = True

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        # operations are performed on log_s
        s = s / self.tau

        if dummy_row:
            assert s.shape[2] >= s.shape[1]
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            ori_nrows = nrows
            nrows = ncols
            s = torch.cat((s, torch.full(dummy_shape, -float('inf')).to(s.device)), dim=1)
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
                s[b, nrows[b]:, :] = -float('inf')
                s[b, :, ncols[b]:] = -float('inf')

        if self.batched_operation:
            log_s = s

            for i in range(self.max_iter):
                if i % 2 == 0:
                    log_sum = torch.logsumexp(log_s, 2, keepdim=True)
                    log_s = log_s - log_sum
                    log_s[torch.isnan(log_s)] = -float('inf')
                else:
                    log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                    log_s = log_s - log_sum
                    log_s[torch.isnan(log_s)] = -float('inf')

                # ret_log_s[b, row_slice, col_slice] = log_s

            if dummy_row and dummy_shape[1] > 0:
                log_s = log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if matrix_input:
                log_s.squeeze_(0)

            return torch.exp(log_s)
        else:
            ret_log_s = torch.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

            for b in range(batch_size):
                row_slice = slice(0, nrows[b])
                col_slice = slice(0, ncols[b])
                log_s = s[b, row_slice, col_slice]

                for i in range(self.max_iter):
                    if i % 2 == 0:
                        log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                        log_s = log_s - log_sum
                    else:
                        log_sum = torch.logsumexp(log_s, 0, keepdim=True)
                        log_s = log_s - log_sum

                ret_log_s[b, row_slice, col_slice] = log_s

            if dummy_row:
                if dummy_shape[1] > 0:
                    ret_log_s = ret_log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if transposed:
                ret_log_s = ret_log_s.transpose(1, 2)
            if matrix_input:
                ret_log_s.squeeze_(0)

            return torch.exp(ret_log_s)

        # ret_log_s = torch.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

        # for b in range(batch_size):
        #    row_slice = slice(0, nrows[b])
        #    col_slice = slice(0, ncols[b])
        #    log_s = s[b, row_slice, col_slice]

    def forward_ori(self, s, nrows=None, ncols=None, dummy_row=False):
        r"""
        Computing sinkhorn with row/column normalization.
        .. warning::
            This function is deprecated because :meth:`~src.lap_solvers.sinkhorn.Sinkhorn.forward_log` is more
            numerically stable.
        """
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        #s = s.to(dtype=dtype)

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        # tau scaling
        ret_s = torch.zeros_like(s)
        for b, n in enumerate(nrows):
            ret_s[b, 0:n, 0:ncols[b]] = \
                nn.functional.softmax(s[b, 0:n, 0:ncols[b]] / self.tau, dim=-1)
        s = ret_s

        # add dummy elements
        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            #s = torch.cat((s, torch.full(dummy_shape, self.epsilon * 10).to(s.device)), dim=1)
            #nrows = nrows + dummy_shape[1] # non in-place
            s = torch.cat((s, torch.full(dummy_shape, 0.).to(s.device)), dim=1)
            ori_nrows = nrows
            nrows = ncols
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = self.epsilon

        row_norm_ones = torch.zeros(batch_size, s.shape[1], s.shape[1], device=s.device, dtype=s.dtype)  # size: row x row
        col_norm_ones = torch.zeros(batch_size, s.shape[2], s.shape[2], device=s.device, dtype=s.dtype)  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        s += self.epsilon

        for i in range(self.max_iter):
            if i % 2 == 0:
                # column norm
                #ones = torch.ones(batch_size, s.shape[1], s.shape[1], device=s.device)
                sum = torch.sum(torch.mul(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2)
            else:
                # row norm
                # ones = torch.ones(batch_size, s.shape[2], s.shape[2], device=s.device)
                sum = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), dim=2)

            tmp = torch.zeros_like(s)
            for b in range(batch_size):
                row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
                col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        if dummy_row:
            if dummy_shape[1] > 0:
                s = s[:, :-dummy_shape[1]]
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = 0

        if matrix_input:
            s.squeeze_(0)

        return s





import torch
import torch.nn as nn
from torch import Tensor
# from src.utils.config import cfg


def soft_topk(scores, ks, max_iter=10, tau=1., nrows=None, ncols=None, return_prob=False):
    """
    Topk-GM algorithm to suppress matches containing outliers.

    :param scores: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
    :param ks: :math:`(b)` number of matches of each graph pair
    :param max_iter: maximum iterations (default: ``10``)
    :param tau: the hyper parameter :math:`\tau` in Sinkhorn algorithm (default: ``1``)
    :param nrows: :math:`(b)` number of objects in dim1
    :param ncols: :math:`(b)` number of objects in dim2
    :param return_prob: whether to return the soft permutation matrix (default: ``False``)
    :return: :math:`(b\times n_1 \times n_2)` the hard permutation matrix
              if ``return_prob=True``, also return :math:`(b\times n_1 \times n_2)` the computed soft permutation matrix
    """
    dist_mat_list = []
    for idx in range(scores.shape[0]):
        n1 = nrows[idx]
        n2 = ncols[idx]
        anchors = torch.tensor([scores[idx, 0: n1, 0: n2].min(), scores[idx, 0: n1, 0: n2].max()], device=scores.device)
        single_dist_mat = -torch.abs(
            scores[idx, 0: n1, 0: n2].reshape(-1).unsqueeze(-1) - anchors.unsqueeze(0))  # .view(n1, n2, 2)
        dist_mat_list.append(single_dist_mat)

    row_prob = torch.ones(scores.shape[0], scores.shape[1] * scores.shape[2], device=scores.device)
    col_prob = torch.zeros((scores.shape[0], 2), dtype=torch.float, device=scores.device)
    col_prob[:, 1] += ks
    col_prob[:, 0] += nrows * ncols - ks

    sk = Sinkhorn_m(max_iter=max_iter, tau=tau, batched_operation=False)

    output = sk(dist_mat_list, row_prob, col_prob, nrows, ncols)

    top_indices = torch.argsort(output[:, :, 1], descending=True, dim=-1)

    output_s = torch.full(scores.shape, 0, device=scores.device, dtype=scores.dtype)
    for batch in range(output_s.shape[0]):
        output_s[batch, 0: nrows[batch], 0: ncols[batch]] = output[batch, 0: nrows[batch] * ncols[batch], 1].view(nrows[batch], -1)

    x = torch.zeros(scores.shape, device=scores.device)
    x = greedy_perm(x, top_indices, ks)

    if return_prob:
        return x, output_s
    else:
        return x


def greedy_perm(x, top_indices, ks):
    """
    Greedy-topk algorithm to select matches with topk confidences.

    :param x: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
    :param top_indices: indices of topk matches
    :param ks: :math:`(b)` number of matches of each graph pair
    :return: :math:`(b\times n_1 \times n_2)` the hard permutation matrix retaining only topk matches
    """
    for b in range(x.shape[0]):
        matched = 0
        cur_idx = 0
        reference_matched_num = round(ks[b].item())
        # if cfg.DATASET_FULL_NAME == 'PascalVOC':
        #     if 'afat-i' in cfg.MODEL_NAME:
        #         reference_matched_num = torch.floor(ks[b])
        #     if 'afat-u' in cfg.MODEL_NAME:
        #         reference_matched_num = torch.ceil(ks[b])
        # if cfg.DATASET_FULL_NAME == 'SPair71k':
        #     if 'afat-u' in cfg.MODEL_NAME:
        reference_matched_num = torch.ceil(ks[b])
        while matched < reference_matched_num and cur_idx < top_indices.shape[1]:  # torch.ceil(n_points[b])
            idx = top_indices[b][cur_idx]
            row = idx // x.shape[2]  # row = torch.div(idx, x.shape[2], rounding_mode='floor')
            col = idx % x.shape[2]
            if x[b, :, col].sum() < 1 and x[b, row, :].sum() < 1:
                x[b, row, col] = 1
                matched += 1
            cur_idx += 1
    return x


class Sinkhorn_m(nn.Module):
    r"""
    Sinkhorn algorithm with marginal distributions turns the input matrix to satisfy the marginal distributions.

    Sinkhorn algorithm firstly applies an ``exp`` function with temperature :math:`\tau`:

    .. math::
        \mathbf{\Gamma}_{i,j} = \exp \left(\frac{\mathbf{\gamma}_{i,j}}{\tau}\right)

    And then turns the matrix into doubly-stochastic matrix by iterative row- and column-wise normalization:

    .. math::
        \mathbf{\Gamma} &= \text{diag}\left((\mathbf{\Gamma} \mathbf{1} \oslash \mathbf{r})\right)^{-1} \mathbf{\Gamma}\\
        \mathbf{\Gamma} &= \text{diag}\left((\mathbf{\Gamma}^{\top} \mathbf{1} \oslash \mathbf{c})\right)^{-1} \mathbf{\Gamma}

    where :math:`\oslash` means element-wise division, :math:`\mathbf{1}` means a column-vector
    whose elements are all :math:`1`\ s, :math:`\mathbf{r}` and :math:`\mathbf{c}` refers to row and column distribution, respectively.

    :param max_iter: maximum iterations (default: ``10``)
    :param tau: the hyper parameter :math:`\tau` controlling the temperature (default: ``1``)
    :param epsilon: a small number for numerical stability (default: ``1e-4``)
    :param log_forward: apply log-scale computation for better numerical stability (default: ``True``)
    :param batched_operation: apply batched_operation for better efficiency (but may cause issues for back-propagation,
     default: ``False``)

    .. note::
        ``tau`` is an important hyper parameter to be set for Sinkhorn algorithm. ``tau`` controls the distance between
        the predicted doubly-stochastic matrix, and the discrete permutation matrix computed by Hungarian algorithm (see
        :func:`~src.lap_solvers.hungarian.hungarian`). Given a small ``tau``, Sinkhorn performs more closely to
        Hungarian, at the cost of slower convergence speed and reduced numerical stability.

    .. note::
        We recommend setting ``log_forward=True`` because it is more numerically stable. It provides more precise
        gradient in back propagation and helps the model to converge better and faster.

    .. warning::
        If you set ``log_forward=False``, this function behaves a little bit differently: it does not include the
        ``exp`` part.

    .. note::
        Setting ``batched_operation=True`` may be preferred when you are doing inference with this module and do not
        need the gradient.
    """

    def __init__(self, max_iter: int = 10, tau: float = 1., epsilon: float = 1e-4,
                 log_forward: bool = True, batched_operation: bool = False):
        super(Sinkhorn_m, self).__init__()
        self.max_iter = max_iter
        self.tau = tau
        self.epsilon = epsilon
        self.log_forward = log_forward
        if not log_forward:
            print('Warning: Sinkhorn algorithm without log forward is deprecated because log_forward is more stable.')
        self.batched_operation = batched_operation  # batched operation may cause instability in backward computation,
        # but will boost computation.

    def forward(self, s: Tensor, row_prob: Tensor, col_prob: Tensor, nrows: Tensor = None, ncols: Tensor = None,
                dummy_row: bool = False) -> Tensor:
        r"""
        :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
        :param row_prob: marginal distribution for row elements
        :param col_prob: marginal distribution for column elements
        :param nrows: :math:`(b)` number of objects in dim1
        :param ncols: :math:`(b)` number of objects in dim2
        :param dummy_row: whether to add dummy rows (rows whose elements are all 0) to pad the matrix to square matrix.
         default: ``False``
        :return: :math:`(b\times n_1 \times n_2)` the computed doubly-stochastic matrix

        .. note::
            We support batched instances with different number of nodes, therefore ``nrows`` and ``ncols`` are
            required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
            the batched matrices are not padded.

        .. note::
            The original Sinkhorn algorithm only works for square matrices. To handle cases where the graphs to be
            matched have different number of nodes, it is a common practice to add dummy rows to construct a square
            matrix. After the row and column normalizations, the padded rows are discarded.

        .. note::
            We assume row number <= column number. If not, the input matrix will be transposed.
        """
        if self.log_forward:
            return self.forward_log(s, row_prob, col_prob, nrows, ncols, dummy_row)
        else:
            raise NotImplementedError

    def forward_log(self, s, row_prob, col_prob, nrows=None, ncols=None, dummy_row=True):
        """Compute sinkhorn with row/column normalization in the log space."""
        # if len(s.shape) == 2:
        #     s = s.unsqueeze(0)
        #     matrix_input = True
        # elif len(s.shape) == 3:
        #     matrix_input = False
        # else:
        #     raise ValueError('input data shape not understood.')
        matrix_input = False

        batch_size = len(s)  # s.shape[0]

        # operations are performed on log_s
        s = [s[i] / self.tau for i in range(len(s))]

        log_row_prob = torch.log(row_prob).unsqueeze(2)
        log_col_prob = torch.log(col_prob).unsqueeze(1)

        if self.batched_operation:
            log_s = s
            last_log_s = log_s

            for i in range(self.max_iter):
                if i % 2 == 0:
                    log_sum = torch.logsumexp(log_s, 2, keepdim=True)
                    log_s = log_s - log_sum + log_row_prob
                    log_s[torch.isnan(log_s)] = -float('inf')
                    # print(i, torch.max(torch.norm((log_s - last_log_s).view(batch_size, -1), dim=-1)), torch.mean(torch.norm((log_s - last_log_s).view(batch_size, -1), dim=-1)))
                    if torch.max(torch.norm((log_s - last_log_s).view(batch_size, -1), dim=-1)) <= 1e-2:
                        # print(i)
                        break
                    last_log_s = log_s
                else:
                    log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                    log_s = log_s - log_sum + log_col_prob
                    log_s[torch.isnan(log_s)] = -float('inf')

                # ret_log_s[b, row_slice, col_slice] = log_s

            # if i == self.max_iter - 1:
            # print('warning: Sinkhorn is not converged.')

            if matrix_input:
                log_s.squeeze_(0)

            return torch.exp(log_s)
        else:
            # ret_log_s = torch.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)
            ret_log_s = torch.full((batch_size, nrows.max() * ncols.max(), 2), -float('inf'), device=s[0].device, dtype=s[0].dtype)

            for b in range(batch_size):
                # row_slice = slice(0, nrows[b])
                # col_slice = slice(0, ncols[b])
                log_s = s[b]

                for i in range(self.max_iter):
                    if i % 2 == 0:
                        log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                        log_s = log_s - log_sum + log_row_prob[b, 0: nrows[b] * ncols[b]]
                        log_s[torch.isnan(log_s)] = -float('inf')
                    else:
                        log_sum = torch.logsumexp(log_s, 0, keepdim=True)
                        log_s = log_s - log_sum + log_col_prob[b]
                        log_s[torch.isnan(log_s)] = -float('inf')
                step = self.max_iter
                while torch.any(log_s > 0):
                    if step % 2 == 0:
                        log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                        log_s = log_s - log_sum + log_row_prob[b, 0: nrows[b] * ncols[b]]
                        log_s[torch.isnan(log_s)] = -float('inf')
                    else:
                        log_sum = torch.logsumexp(log_s, 0, keepdim=True)
                        log_s = log_s - log_sum + log_col_prob[b]
                        log_s[torch.isnan(log_s)] = -float('inf')
                    step += 1

                ret_log_s[b, 0: nrows[b] * ncols[b]] = log_s
            # if dummy_row:
            #     if dummy_shape[1] > 0:
            #         ret_log_s = ret_log_s[:, :-dummy_shape[1]]
            #     for b in range(batch_size):
            #         ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')
            #
            # if transposed:
            #     ret_log_s = ret_log_s.transpose(1, 2)
            if matrix_input:
                ret_log_s.squeeze_(0)

            return torch.exp(ret_log_s)

        # ret_log_s = torch.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

        # for b in range(batch_size):
        #    row_slice = slice(0, nrows[b])
        #    col_slice = slice(0, ncols[b])
        #    log_s = s[b, row_slice, col_slice]

    def forward_ori(self, s, nrows=None, ncols=None, dummy_row=False):
        r"""
        Computing sinkhorn with row/column normalization.

        .. warning::
            This function is deprecated because :meth:`~src.lap_solvers.sinkhorn.Sinkhorn.forward_log` is more
            numerically stable.
        """
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        # s = s.to(dtype=dtype)

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        # tau scaling
        ret_s = torch.zeros_like(s)
        for b, n in enumerate(nrows):
            ret_s[b, 0:n, 0:ncols[b]] = \
                nn.functional.softmax(s[b, 0:n, 0:ncols[b]] / self.tau, dim=-1)
        s = ret_s

        # add dummy elements
        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            # s = torch.cat((s, torch.full(dummy_shape, self.epsilon * 10).to(s.device)), dim=1)
            # nrows = nrows + dummy_shape[1] # non in-place
            s = torch.cat((s, torch.full(dummy_shape, 0.).to(s.device)), dim=1)
            ori_nrows = nrows
            nrows = ncols
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = self.epsilon

        row_norm_ones = torch.zeros(batch_size, s.shape[1], s.shape[1], device=s.device,
                                    dtype=s.dtype)  # size: row x row
        col_norm_ones = torch.zeros(batch_size, s.shape[2], s.shape[2], device=s.device,
                                    dtype=s.dtype)  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        s += self.epsilon

        for i in range(self.max_iter):
            if i % 2 == 0:
                # column norm
                # ones = torch.ones(batch_size, s.shape[1], s.shape[1], device=s.device)
                sum = torch.sum(torch.mul(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2)
            else:
                # row norm
                # ones = torch.ones(batch_size, s.shape[2], s.shape[2], device=s.device)
                sum = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), dim=2)

            tmp = torch.zeros_like(s)
            for b in range(batch_size):
                row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
                col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        if dummy_row:
            if dummy_shape[1] > 0:
                s = s[:, :-dummy_shape[1]]
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = 0

        if matrix_input:
            s.squeeze_(0)

        return s


if __name__ == '__main__':
    # bs = Sinkhorn_m(max_iter=8, epsilon=1e-4)
    # inp = torch.tensor([[[1., 0, 1.],
    #                      [1., 0, 3.],
    #                      [2., 0, 1.],
    #                      [4., 0, 2.]]], requires_grad=True)
    # outp = bs(inp, (3, 4))
    #
    # print(outp)
    # l = torch.sum(outp)
    # l.backward()
    # print(inp.grad * 1e10)
    #
    # outp2 = torch.tensor([[0.1, 0.1, 1],
    #                       [2, 3, 4.]], requires_grad=True)
    #
    # l = torch.sum(outp2)
    # l.backward()
    # print(outp2.grad)

    scores = torch.tensor([[[0.2, 0.5, 0.3, 0.1],
                           [0.4, 0.1, 0.8, 0.6]]])
    ks = torch.tensor([2, 3])
    nrows=scores.size(0)
    ncols= scores.size(1)

    # 调用函数
    output = soft_topk(scores, ks, max_iter=10, tau=1., nrows=nrows, ncols=ncols, return_prob=False)
    print(output)
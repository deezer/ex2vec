import torch

from engine import Engine
from utils import resume_checkpoint, use_cuda


# Ex2Vec Class
class Ex2Vec(torch.nn.Module):
    def __init__(self, config):
        super(Ex2Vec, self).__init__()
        self.config = config
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.latent_d = config["latent_dim"]

        # lambda parameter to move the embedding
        self.global_lamb = torch.nn.Parameter(torch.tensor(1.0))

        self.user_lamb = torch.nn.Embedding(self.n_users, 1)
        # self.item_lamb = torch.nn.Embedding(self.n_items, 1)

        self.user_bias = torch.nn.Embedding(self.n_users, 1)
        self.item_bias = torch.nn.Embedding(self.n_items, 1)

        # quadratic function parameters
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.beta = torch.nn.Parameter(torch.tensor(-0.065))
        self.gamma = torch.nn.Parameter(torch.tensor(0.5))

        # the cutoff value
        self.cutoff = torch.nn.Parameter(torch.tensor(3.0))

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_d
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_d
        )

        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices, r_interval):
        user_embeddings = self.embedding_user(user_indices)  # + 10
        item_embeddings = self.embedding_item(item_indices)  # + 10

        u_bias = self.user_bias(user_indices).squeeze(-1)
        i_bias = self.item_bias(item_indices).squeeze(-1)

        difference = torch.sub(item_embeddings, user_embeddings)
        base_distance = torch.sqrt((difference**2)).sum(axis=1)

        # compute the base_level activation
        # get only time gaps superior to zero
        mask = (r_interval > 0).float()
        delta_t = r_interval * mask

        delta_t = delta_t + self.cutoff.clamp(min=0.1, max=100)
        decay = 0.5  # self.decay.clamp(min = 0.01, max = 10)
        base_level = torch.sum(torch.pow(delta_t, -decay) * mask, 1)

        # compute how much to move the user embedding
        lamb = self.global_lamb.clamp(0.01, 10) + self.user_lamb(user_indices).squeeze(-1).clamp(0.1, 10)

        base_activation = torch.mul(base_level, lamb)

        activation = torch.minimum(base_activation, base_distance)
        # move the user embedding in the direction of the item given a factor lambda
        distance = base_distance - activation  # self.lamb*distance*base_level
        # apply the quadratic funcion
        I = self.alpha * distance  + self.beta * torch.pow(distance, 2) + self.gamma + u_bias + i_bias

        # output the interest value
        interest = self.logistic(I)
        return interest

    def load_pretrain_weights(self):
        """Loading weights from trained GMF model"""
        config = self.config

        ex2vec_pre = Ex2Vec(config)
        if config["use_cuda"] is True:
            ex2vec_pre.cuda()

            resume_checkpoint(
                ex2vec_pre,
                model_dir=config["pretrain_dir"],
                device_id=config["device_id"],
            )

        # embeddings
        self.embedding_user.weight.data = ex2vec_pre.embedding_user.weight.data
        self.embedding_item.weight.data = ex2vec_pre.embedding_item.weight.data

    def init_weight(self):
        pass


class Ex2VecEngine(Engine):
    """Engine for training & evaluating MEE model"""

    def __init__(self, config):
        self.model = Ex2Vec(config)
        if config["use_cuda"] is True:
            use_cuda(True, config["device_id"])
            self.model.cuda()
        super(Ex2VecEngine, self).__init__(config)
        print(self.model)
        for name, param in self.model.named_parameters():
            print(name, type(param.data), param.size())

        if config["pretrain"]:
            self.model.load_pretrain_weights()

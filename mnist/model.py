from imports import *
from Pythia.config import *

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        weight = torch.empty(5,5)
        ortho_weight = nn.init.orthogonal_(weight)
        self.ortho_weight = torch.nn.Parameter(ortho_weight)
    
    def forward(self, x):
        product = torch.matmul(x, self.ortho_weight.T)
        return product 
    

class mnistmodel(nn.Module):
    def __init__(self):
        super(mnistmodel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)



    
class RotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, init_orth=True):
        super().__init__()
        weight = torch.empty(n, n)
        if init_orth:
            torch.nn.init.orthogonal_(weight)
        self.weight = torch.nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


def dasmat(model_path):
        
    mnist_model = mnistmodel()

    mnist_model.load_state_dict(torch.load(model_path))

    # freezing the whole trained model.
    def require_grad(model, grad = False):
        for param in model.parameters():
            param.requires_grad = grad


    def gumbel_sigmoid(x, tau=0.5):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(x)))
        y = torch.sigmoid((x + gumbel_noise) / tau)
        return (y > 0.5).float() - y.detach() + y



    require_grad(mnist_model, False)

    class new_model(nn.Module):
        def __init__(self, model):
            super(new_model, self).__init__()
            self.conv1 = model.conv1
            self.conv2 = model.conv2
            self.conv2_drop = model.conv2_drop
            self.fc1 = model.fc1
            self.fc2 = model.fc2
            self.fc3 = model.fc3
            input_shape1 = self.fc1.out_features
            input_shape2 = self.fc2.out_features
            rotate_layer1 = RotateLayer(input_shape1)
            rotate_layer2 = RotateLayer(input_shape2)
            self.rotate_layer_1 = torch.nn.utils.parametrizations.orthogonal(rotate_layer1)
            self.rotate_layer_2 = torch.nn.utils.parametrizations.orthogonal(rotate_layer2)
            # bm = torch.nn.init.xavier_uniform_(torch.empty(1,input_shape))
            bm1 = torch.zeros(1, input_shape1)

            # self.transpose_matrix = torch.nn.Parameter(ortho_fixed_weight)
            binary_mask1 = torch.nn.Parameter(bm1, requires_grad=True)
            self.binary_mask1 = gumbel_sigmoid(binary_mask1)
            
            bm2 = torch.zeros(1, input_shape2)
            binary_mask2 = torch.nn.Parameter(bm2, requires_grad=True)
            self.binary_mask2 = gumbel_sigmoid(binary_mask2)
            # assert input_shape == output_shape
            # self.ones_matrix = torch.nn.Parameter(torch.eye(input_shape), requires_grad=True)
            # w = torch.empty(input_shape, output_shape)
            # self.transpose_matrix = nn.Parameter(nn.init.xavier_uniform_(w), 
                                                #  requires_grad=True)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            
            # rotating
            x = self.rotate_layer_1(x)

            # masking
            b = self.binary_mask1.repeat(x.shape[0], 1)
            b = b.to(x.device)
            x = torch.mul(x,b) 
            
            #inversing the rotating operation
            x = torch.matmul(x, self.rotate_layer_1.weight.T)

            x = F.relu(self.fc2(x))

            # rotating
            x = self.rotate_layer_2(x)

            # masking
            b2 = self.binary_mask2.repeat(x.shape[0], 1)
            b2 = b2.to(x.device)
            x = torch.mul(x,b2) 

            #inversing the rotating operation
            x = torch.matmul(x, self.rotate_layer_2.weight.T)

            x = self.fc3(x)
            return F.log_softmax(x, dim = 1)
    
    new_model = new_model(mnist_model)

    return new_model


###---------------------------------------Building Sparse Autoencoder------------------------------###
@dataclass
class AutoEncoderConfig:
    n_instances: int
    # n_input_ae: int
    # n_hidden_ae: int
    l1_coeff: float = 0.0001
    tied_weights: bool = True

'''
We are doing this for the image domain so we will not be considering the n_instances shape
'''


class AutoEncoder(nn.Module):
    W_enc: Float[Tensor, "n_input_ae n_hidden_ae"]
    W_dec: Float[Tensor, "n_hidden_ae n_input_ae"]
    b_enc: Float[Tensor, "n_hidden_ae"]
    b_dec: Float[Tensor, "n_input_ae"]


    def __init__(self, cfg: AutoEncoderConfig, n_input_ae, n_hidden_ae):
        super().__init__()

        def gumbel_sigmoid(x, tau=0.5):
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(x)))
            y = torch.sigmoid((x + gumbel_noise) / tau)
            return (y > 0.5).float() - y.detach() + y
        
        self.cfg = cfg
        cfg.n_input_ae = n_input_ae
        cfg.n_hidden_ae = n_hidden_ae
        self.W_enc = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_input_ae, cfg.n_hidden_ae))),
                                  requires_grad=True)
        if not(cfg.tied_weights):
            self.W_dec = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_hidden_ae, cfg.n_input_ae))))
        self.b_enc = nn.Parameter(t.zeros(cfg.n_hidden_ae), requires_grad=True)
        self.b_dec = nn.Parameter(t.zeros(cfg.n_input_ae), requires_grad=True)
        self.to(device)

        # bm = torch.nn.init.xavier_uniform_(torch.empty(1,self.cfg.n_hidden_ae))
        # binary_mask1 = torch.nn.Parameter(bm, requires_grad=True)
        # self.binary_mask1 = gumbel_sigmoid(binary_mask1)


    def forward(self, h: Float[Tensor, "batch_size n_hidden"]):

        # Compute activations
        h_cent = h - self.b_dec
        # print(f"Shape of h_cent: {h_cent.shape}")
        # print(f"Shape of W_enc: {self.W_enc.shape}")
        acts = einops.einsum(
            h_cent, self.W_enc,
            "batch_size n_input_ae, n_input_ae n_hidden_ae -> batch_size n_hidden_ae"
        )
        acts = F.relu(acts + self.b_enc)
        # acts = self.binary_mask1 * acts 

        # Compute reconstructed input
        # h_reconstructed = einops.einsum(
        #     acts, (self.W_enc.T if self.cfg.tied_weights else self.W_dec),
        #     "batch_size n_hidden_ae, n_hidden_ae n_input_ae -> batch_size n_input_ae"
        # ) + self.b_dec
        h_reconstructed = einops.einsum(
            acts, (self.W_enc.T),
            "batch_size n_hidden_ae, n_hidden_ae n_input_ae -> batch_size n_input_ae"
        ) + self.b_dec

        # Compute loss, return values
        l2_loss = (h_reconstructed - h).pow(2).mean() # shape [batch_size n_instances]
        l1_loss = acts.abs().sum() # shape [batch_size n_instances]
        # loss = (self.cfg.l1_coeff * l1_loss + l2_loss).mean(0).sum() # scalar
        # print("The value of cfg.l1_coeff is: ", self.cfg.l1_coeff)
        loss = self.cfg.l1_coeff * l1_loss + l2_loss

        return l1_loss, l2_loss, loss, acts, h_reconstructed


    @t.no_grad()
    def normalize_decoder(self) -> None:
        '''
        Normalizes the decoder weights to have unit norm. If using tied weights, we we assume W_enc is used for both.
        '''
        if self.cfg.tied_weights:
            self.W_enc.data = self.W_enc.data / self.W_enc.data.norm(dim=1, keepdim=True)
        else:
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=2, keepdim=True)


class sparse_encoder(nn.Module):
    
    def __init__(self, model_path, ae_cfg: AutoEncoderConfig):
        super(sparse_encoder, self).__init__()
        self.mnist_model = mnistmodel()

        self.mnist_model.load_state_dict(torch.load(model_path))

        # freezing the whole trained model.
        def require_grad(model, grad = False):
            for param in model.parameters():
                param.requires_grad = grad

        require_grad(self.mnist_model, False)

        self.conv1 = self.mnist_model.conv1
        self.conv2 = self.mnist_model.conv2
        self.conv2_drop = self.mnist_model.conv2_drop
        self.fc1 = self.mnist_model.fc1
        self.fc2 = self.mnist_model.fc2
        self.fc3 = self.mnist_model.fc3
        input_shape1 = self.fc1.out_features
        input_shape2 = self.fc2.out_features
        self.sae = AutoEncoder(ae_cfg, input_shape1, 5*input_shape1)
        

    def forward(self, x):


        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))

        # sae
        l1_loss, l2_loss, loss, acts, x_ = self.sae(x)

        x = self.fc2(x_)
        x = self.fc3(x)
        return l1_loss, l2_loss, loss, acts, F.log_softmax(x, dim = 1)
    

class mask_sparse_encoder(nn.Module):
    
    def __init__(self, model_path, mnist_model_path, ae_cfg: AutoEncoderConfig):
        super(mask_sparse_encoder, self).__init__()

        self.sae_model = sparse_encoder(mnist_model_path, ae_cfg)

        self.sae_model.load_state_dict(torch.load(model_path))


        def gumbel_sigmoid(x, tau=0.5):
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(x)))
            y = torch.sigmoid((x + gumbel_noise) / tau)
            return (y > 0.5).float() - y.detach() + y

        # freezing the whole trained model.
        def require_grad(model, grad = False):
            for param in model.parameters():
                param.requires_grad = grad

        require_grad(self.sae_model, False)

        self.conv1 = self.sae_model.conv1
        self.conv2 = self.sae_model.conv2
        self.conv2_drop = self.sae_model.conv2_drop
        self.fc1 = self.sae_model.fc1
        self.fc2 = self.sae_model.fc2
        self.fc3 = self.sae_model.fc3
        input_shape1 = self.fc1.out_features
        input_shape2 = self.fc2.out_features
        self.enc = self.sae_model.sae.W_enc
        self.b_enc = self.sae_model.sae.b_enc
        self.b_dec = self.sae_model.sae.b_dec
        self.sae = AutoEncoder(ae_cfg, input_shape1, 5*input_shape1)
        self.cfg = ae_cfg
        
        bm1 = torch.zeros(1, 5*input_shape1)
        # self.transpose_matrix = torch.nn.Parameter(ortho_fixed_weight)
        self.binary_mask = torch.nn.Parameter(bm1, requires_grad=True)
        # self.binary_mask1 = gumbel_sigmoid(binary_mask)

    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        h = F.relu(self.fc1(x))

        # integrating the sparse autoencoder
        h_cent = h - self.b_dec
        acts = einops.einsum(
            h_cent, self.enc,
            "batch_size n_input_ae, n_input_ae n_hidden_ae -> batch_size n_hidden_ae"
        )
        acts = F.relu(acts + self.b_enc)

        #inserting the mask in it
        acts = self.binary_mask * acts

        h_reconstructed = einops.einsum(
            acts, (self.enc.T),
            "batch_size n_hidden_ae, n_hidden_ae n_input_ae -> batch_size n_input_ae"
        ) + self.b_dec

        # Compute loss, return values
        l2_loss = (h_reconstructed - h).pow(2).mean() # shape [batch_size n_instances]
        l1_loss = acts.abs().sum() # shape [batch_size n_instances]
        loss = self.cfg.l1_coeff * l1_loss + l2_loss

        x = self.fc2(h_reconstructed)
        x = self.fc3(x)
        return l1_loss, l2_loss, loss, acts, F.log_softmax(x, dim = 1)
    



# ------------------------ Making the sparse autoencoder with mask``` ------------------------ #
'''
We will be taking the pre-trained weights as the mask error and the reconstruction should contradict 
each other, as a result, affecting the learning of the model.
'''


if __name__ == '__main__':

    '''
    So our objective would be to build the model that transform the 
    matrix of the input to an orthogonal matrix such that it becomes 
    exactly equal to the input variable passed. 
    '''
    ae_cfg = AutoEncoderConfig(n_instances = batch_size_train,
                                l1_coeff = 0.2,
                                tied_weights = False
                                )

    # model = Model()
    # print(model)
    # mnist_model = mnistmodel()
    # print(mnist_model)

    # model = dasmat("models/Mon_Apr_29_00:26:24_2024.pt")
    model = sparse_encoder("mnist_models/2_0133.pt", ae_cfg)
    for p in model.parameters():
        if p.requires_grad:
            print(p.name, p.data)
    # print(model)


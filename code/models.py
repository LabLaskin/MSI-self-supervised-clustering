#===================================
# 1. pretrained models: 1. effnet
#===================================
# 1) effnet, extract embeddings at fc0
class effnet_model_fc0(nn.Module):
    class Identity(nn.Module):
        def __init__(self): super().__init__()
        def forward(self, x):
            return x
    def __init__(self):
        super().__init__()
        base_model = EfficientNet.from_pretrained('efficientnet-b0')
        base_model._fc = effnet_model_fc0.Identity()
        self.embedding = base_model
    def forward(self, x):
        embedding = self.embedding(x)
        return embedding
        
#===================================
# 2. Contrastive learning: SimCLR
# 2.1) Dataset: with augmentation & pairing
# 2.2) Loss function: contrastive loss
# 2.3) SimCLR model: based on a pretrained model
# 2.4) SimCLR train
#===================================

# 2.1) dataset wrap (Dataset class) to return 2 img augmentations from 1 img
#===================================
class DatasetWrapper_SimCLR(Dataset):
    def __init__(self, ds: Dataset, trans: random):
        super().__init__()
        self.ds = ds
        self.trans = trans
        
        # Normalization for pretrained effnet
        self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.ds)
    
    def __getitem_internal__(self, idx, preprocess=True):
        image_raw = self.ds[idx]
        
        # take 2 separate random augmentations
        t1 = self.trans(image_raw)
        t2 = self.trans(image_raw)
        
        # step 2 preprocess on tensor images.
        if preprocess:
            t1 = self.preprocess(t1)
            t2 = self.preprocess(t2)
        else:
            t1 = transforms.ToTensor()(t1)
            t2 = transforms.ToTensor()(t2)

        return t1, t2   # returns a pair of images and a scalar of 0...

    def __getitem__(self, idx):                       
        return self.__getitem_internal__(idx, True)
    
    def raw(self, idx):
        return self.__getitem_internal__(idx, False)

# 2.2) define the loss function, key
#===================================
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size 
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

# 2.3) SimCLR model
#===================================
class CLR(nn.Module):       
    class Identity(nn.Module):
        def __init__(self): super().__init__()

        def forward(self, x):
            return x
    
        
    def __init__(self, embedding_size=1024):
        super().__init__()
        
        base_model = EfficientNet.from_pretrained("efficientnet-b0")
        internal_embedding_size = base_model._fc.in_features
        base_model._fc = self.Identity() 
        
        # effnet output
        self.embedding = base_model

        # w2*sigma(w1ho)
        self.projection = nn.Sequential(
            nn.Linear(in_features=internal_embedding_size, out_features=internal_embedding_size),       
            nn.ReLU(),                                                                                
            nn.Linear(in_features=internal_embedding_size, out_features=embedding_size)                  
        )

    def forward(self, X):
        image = X
        embedding = self.embedding(image)
        projection = self.projection(embedding)
        return embedding, projection

# 2.4) SimCLR train
#===================================
def get_CLR_training_components(hparams_CLR):
    '''
    input: hparams_CLR
    output: model on cuda, criterion on cuda, optimizer, scheculer, dataloader. 
    '''
    # model
    model_CLR = CLR()
    model_CLR = model_CLR.cuda()
    # criterion
    criterion_CLR = ContrastiveLoss(hparams_CLR.batch_size)
    criterion_CLR = criterion_CLR.cuda()                    # it has initialization, put on cuda
    # optimizer and scheduler
    optimizer_CLR = Adam(model_CLR.parameters(), lr = hparams_CLR.lr)
    scheduler_CLR = CosineAnnealingLR(optimizer_CLR, hparams_CLR.epochs)
    # dataloader
    dataloader_CLR = DataLoader(DatasetWrapper_SimCLR(hparams_CLR.data, hparams_CLR.trans),
                                batch_size = hparams_CLR.batch_size, 
                                num_workers = cpu_count(),                         
                                sampler = SubsetRandomSampler(list(range(hparams_CLR.train_size))),
                                pin_memory = True,
                                drop_last = True)
    return model_CLR, criterion_CLR, optimizer_CLR, scheduler_CLR, dataloader_CLR

def train_CLR(model, criterion, optimizer, dataloader, verbose = True):
    losses = []
    if verbose:
        for i, batch in tqdm(enumerate(dataloader)):
            X, Y = batch
            X = X.cuda(non_blocking=True)
            Y = Y.cuda(non_blocking=True)
            _, proj_X = model(X)
            _, proj_Y = model(Y)
            loss = criterion(proj_X, proj_Y)

            # record
            print('batch {} loss: {}'.format(i, loss))
            losses.append(loss)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        for i, batch in enumerate(dataloader):
            X, Y = batch
            X = X.cuda(non_blocking=True)
            Y = Y.cuda(non_blocking=True)
            _, proj_X = model(X)
            _, proj_Y = model(Y)
            loss = criterion(proj_X, proj_Y)

            # record
            losses.append(loss)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return losses

def fit_CLR(hparams_CLR, output_folder, verbose=False):
    ## get training components with hparams
    model_CLR, criterion_CLR, optimizer_CLR, scheduler_CLR, dataloader_CLR = get_CLR_training_components(hparams_CLR)

    ## main fit steps
    total_losses = []
    for epoch in tqdm(range(hparams_CLR.epochs)):
        # 1. train
        losses = train_CLR(model_CLR, criterion_CLR, optimizer_CLR, dataloader_CLR, verbose=False)      # one buttom to control
        # update lr
        scheduler_CLR.step()
        # 2. record 
        losses_np = torch.tensor(losses).cpu().numpy()
        epoch_loss = np.mean(losses_np)
        total_losses.append(losses_np)
        if verbose:
            print('\nepoch {} loss: {}\n'.format(epoch, epoch_loss))

    ## save model after traning
    torch.save(model_CLR.state_dict(), output_folder + '/model_CLR.ckpt')

    ## record losses
    total_losses = np.array(total_losses).flatten()
    np.save(output_folder + '/losses_CLR.npy', total_losses)
    if verbose:
        plot_losses(total_losses.shape[0], total_losses, 'iterations', 'loss', 'total losses')
    return model_CLR, total_losses

#===================================
# 3. frozen CNN with a linear classifier: (CLR +) CLS, work for data with (pseudo) labels
# 3.1) Dataset: with labels, without augmentation/pairing
# 3.2) CLR_CLS model: based on CLR
# 3.3) CLS train
#===================================

# 3.1) Dataset
#===================================
class DatasetWrapper_CLS(Dataset):
    '''
    1. link data_set and labels (predicted), 
    2. enables data access by getitem 
    '''
    def __init__(self, ds: Dataset, labels, target_size=(96, 96)):
        super().__init__()
        self.ds = ds  # list of PIL imgs
        self.labels = labels
        self.target_size = target_size
        
        # Normalization, keep consistant with pretext. 
        self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): 
        return len(self.ds)

    def __getitem_internal__(self, idx, preprocess=True):
        img = self.ds[idx]
        
        if preprocess:
            img = self.preprocess(img)
        else:
            img = transforms.ToTensor()(img)

        return img, self.labels[idx]    ## final return

    def __getitem__(self, idx):                       
        return self.__getitem_internal__(idx, True)

    def raw(self, idx):
        return self.__getitem_internal__(idx, False)
        
# 3.2) CLR_CLS model
#===================================
class CLR_CLS(nn.Module):
    '''
    CLR base + a linear classifier
    Would train them together in FineTune
    '''
    def __init__(self, n_classes, freeze_base, embeddings_model_path): # since we are going to train the base, so go with only one head
        super().__init__()
        base_model = CLR()
        base_model.load_state_dict(torch.load(embeddings_model_path))
        
        # take output
        self.embedding = base_model.embedding

        if freeze_base:
            print('During the training of current CLR_CLS model, the CNN base is frozen')
            for param in self.embedding.parameters():
                param.requires_grad = False

        # linear layer to cast features to n_classes space
        self.projection = nn.Linear(in_features=base_model.projection[0].in_features, out_features=n_classes) 

    def forward(self, X, *args):
        embedding = self.embedding(X)
        projection = self.projection(embedding)
        return embedding, projection

# 3.3) CLS train
#===================================
def get_CLS_training_components(hparams_CLS):
    '''
    inputs: hparams_CLS
    outputs: model on cuda, criterion, optimizer, scheduler, dataloader
    '''
    # model
    model_CLS = CLR_CLS(hparams_CLS.n_classes, hparams_CLS.freeze_base, hparams_CLS.embeddings_path)
    model_CLS = model_CLS.cuda()
    # criterion
    criterion_CLS = nn.CrossEntropyLoss()
    # optimizer and scheduler
    optimizer_CLS = Adam(model_CLS.projection.parameters(), lr = hparams_CLS.lr)
    scheduler_CLS = CosineAnnealingLR(optimizer_CLS, hparams_CLS.epochs)
    # dataloader 
    dataloader_CLS = DataLoader(DatasetWrapper_CLS(hparams_CLS.data, hparams_CLS.labels),
                                batch_size = hparams_CLS.batch_size,
                                num_workers = cpu_count(),
                                sampler = SubsetRandomSampler(list(range(hparams_CLS.train_size))),
                                pin_memory = True,
                                drop_last = True)
    return model_CLS, criterion_CLS, optimizer_CLS, scheduler_CLS, dataloader_CLS

def train_CLS(model, criterion, optimizer, dataloader, verbose = True):
    losses = []
    if verbose:
        for i, batch in tqdm(enumerate(dataloader)):
            X, Y = batch
            X = X.cuda(non_blocking=True)
            Y = Y.cuda(non_blocking=True)
            _, proj_X = model(X)
            loss = criterion(proj_X, Y)

            # record
            print('batch {} loss: {}'.format(i, loss))
            losses.append(loss)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        for i, batch in enumerate(dataloader):
            X, Y = batch
            X = X.cuda(non_blocking=True)
            Y = Y.cuda(non_blocking=True)
            _, proj_X = model(X)
            loss = criterion(proj_X, Y)

            # record
            losses.append(loss)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return losses
    
def fit_CLS(hparams_CLR, output_folder, verbose=False):
    ## get training components with hparams
    model_CLS, criterion_CLS, optimizer_CLS, scheduler_CLS, dataloader_CLS = get_CLS_training_components(hparams_CLS)

    ## main fit steps
    total_losses = []
    for epoch in tqdm(range(hparams_CLS.epochs)):
        # 1. train
        losses = train_CLS(model_CLS, criterion_CLS, optimizer_CLS, dataloader_CLS, verbose=False)
        # update lr
        scheduler_CLS.step()
        # 2. record
        losses_np = torch.tensor(losses).cpu().numpy()
        epoch_loss = np.mean(losses_np)
        total_losses.append(losses_np)
        if verbose:
            print('\nepoch {} loss: {}\n'.format(epoch, epoch_loss))

    ## save model after traning
    torch.save(model_CLS.state_dict(), output_folder + '/model_CLS.ckpt')

    ## record losses
    total_losses = np.array(total_losses).flatten()
    np.save(output_folder + '/losses_CLS.npy', total_losses)
    if verbose:
        plot_losses(total_losses.shape[0], total_losses, 'iterations', 'loss', 'total losses')
    return model_CLS, total_losses
    
#===================================
# 4. FineTune
# 4.1) Dataset: with augmentation & pairing
# 4.2) loss function: ConfidenceBasedCE
# 4.X) FineTune model: None, directly use CLR_CLS backbone and weights
# 4.3) FineTune train
#===================================

# 4.1) Dataset
#===================================
class DatasetWrapper_FineTune(Dataset):
    '''
    two modes: 1. me vs myself. 2. me vs 1 of neighbors
    2 trans: 1 for anchor img, 1 for peer img 
    '''
    def __init__(self, ds: Dataset, trans1: random, trans2:random):
        super().__init__()
        self.ds = ds
        self.anchor_trans = trans1
        self.peer_trans = trans2
        
        # Normalization, keep consistant with pretext. 
        self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.ds)
    
    def __getitem_internal__(self, idx, preprocess=True):
        # anchor's weak augmentation vs. anchor's strong augmentation
        anchor_img_raw = self.ds[idx]
        peer_img_raw = self.ds[idx]
            
        t1 = self.anchor_trans(anchor_img_raw)
        t2 = self.peer_trans(peer_img_raw)
        
        # step 2 preprocess on tensor images. so .ToTensor() is a necessary step beforehands. 
        if preprocess:
            t1 = self.preprocess(t1)
            t2 = self.preprocess(t2)
        else:
            t1 = transforms.ToTensor()(t1)
            t2 = transforms.ToTensor()(t2)

        return t1, t2   # returns a pair of images

    def __getitem__(self, idx):                       
        return self.__getitem_internal__(idx, True)
    
    def raw(self, idx):
        return self.__getitem_internal__(idx, False)        
        
# 4.2) loss function
#===================================
class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)

class ConfidenceBasedCE(nn.Module):
    '''
    parameters:
    1. confidence threshold [0, 1] 
    2. apply_class_balancing 
    3. include_all_classes
    '''
    def __init__(self, threshold, apply_class_balancing):
        super().__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold    
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_proj, peers_proj):
        """
        Loss function for fine tuning. Backward focuses on the peer_proj.

        input: logits for anchors and for its peers (their strong augmentations or their neighbors), shape=(n_batch, n_classes)
        output: cross entropy 
        """
        # Retrieve target and mask based on weakly augmentated anchors
        anchors_prob = self.softmax(anchors_proj) 
        max_prob, target = torch.max(anchors_prob, dim = 1)
        mask = max_prob > self.threshold 
        b, c = anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)
        masked_classes, counts = torch.unique(target_masked, return_counts = True)
        n_miss_classes = c - masked_classes.size(0)           # could log it.

        # Inputs are strongly augmented anchors
        input_ = peers_proj

        # Class balancing weights
        if self.apply_class_balancing:
            #idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[masked_classes] = freq

        else:
            weight = None

        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean') 
        
        return loss, n, n_miss_classes, weight
        
# 4.3) FineTune train
#===================================
def get_FineTune_training_components(hparams_FineTune):
    '''
    inputs: hparams_FineTune
    outputs: model on cuda, criterion, optimizer, scheduler, dataloader
    '''
    # model
    model_FineTune = CLR_CLS(hparams_FineTune.n_classes, hparams_FineTune.freeze_base, hparams_FineTune.model_CLR_path) # initialization
    model_FineTune.load_state_dict(torch.load(hparams_FineTune.model_CLS_path))  # load up parameters 
    model_FineTune = model_FineTune.cuda()
    # criterion
    criterion_FineTune = ConfidenceBasedCE(hparams_FineTune.threshold, hparams_FineTune.apply_class_balancing)
    # optimizer and scheduler
    optimizer_FineTune = Adam(model_FineTune.parameters(), lr = hparams_FineTune.lr)
    scheduler_FineTune = CosineAnnealingLR(optimizer_FineTune, hparams_FineTune.epochs)
    # dataloader 
    dataloader_FineTune = DataLoader(DatasetWrapper_FineTune(hparams_FineTune.data, hparams_FineTune.trans1, hparams_FineTune.trans2), 
                          batch_size = hparams_FineTune.batch_size, 
                          num_workers = cpu_count(),
                          sampler = SubsetRandomSampler(list(range(hparams_FineTune.train_size))), 
                          pin_memory = True,
                          drop_last=True)   
    return model_FineTune, criterion_FineTune, optimizer_FineTune, scheduler_FineTune, dataloader_FineTune

def train_FineTune(model, criterion, optimizer, dataloader, verbose = True):
    losses = []
    confidence_fractions = []

    if verbose:
        for i, batch in tqdm(enumerate(dataloader)):
            anchors, peers = batch
            anchors = anchors.cuda(non_blocking=True)
            peers = peers.cuda(non_blocking=True)
            # loss computation based on the prediction
            with torch.no_grad():
                _, anchors_proj = model(anchors)
            _, peers_proj = model(peers)
            loss, n_confidence, _, _ = criterion(anchors_proj, peers_proj)

            # record
            print('batch {} loss: {}'.format(i, loss))
            losses.append(loss)
            confidence_fractions.append(n_confidence/dataloader.batch_size)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        for i, batch in enumerate(dataloader):
            anchors, peers = batch
            anchors = anchors.cuda(non_blocking=True)
            peers = peers.cuda(non_blocking=True)
            # loss computation based on the prediction
            with torch.no_grad():
                _, anchors_proj = model(anchors)
            _, peers_proj = model(peers)
            loss, n_confidence, _, _ = criterion(anchors_proj, peers_proj)

            # record
            losses.append(loss)
            confidence_fractions.append(n_confidence/dataloader.batch_size)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return losses, confidence_fractions

def fit_FineTune(hparams_FineTune, output_folder, verbose=False):
    ## get training components with hparams
    model_FineTune, criterion_FineTune, optimizer_FineTune, scheduler_FineTune, dataloader_FineTune = get_FineTune_training_components(hparams_FineTune)

    ## main fit steps
    total_losses = []
    total_confidence_fractions = []
    for epoch in tqdm(range(hparams_FineTune.epochs)):
        # 1. train
        losses, confidence_fractions = train_FineTune(model_FineTune, criterion_FineTune, optimizer_FineTune, dataloader_FineTune, verbose=False)
        # update lr
        scheduler_FineTune.step()
        # 2. record
        losses_np = torch.tensor(losses).cpu().numpy()
        epoch_loss = np.mean(losses_np)
        total_losses.append(losses_np)
        total_confidence_fractions.append(confidence_fractions)

    ## save model after traning
    torch.save(model_FineTune.state_dict(), output_folder + '/model_FineTune.ckpt')

    ## record losses and confidence fractions
    total_losses = np.array(total_losses).flatten()
    np.save(output_folder + '/losses_FineTune.npy', total_losses)
    total_confidence_fractions = np.array(total_confidence_fractions).flatten()
    np.save(output_folder + '/confidence_fractions_FineTune.npy', total_confidence_fractions)
    if verbose:
        plot_losses(total_losses.shape[0], total_losses, 'iterations', 'loss', 'total losses')
        plot_losses(total_confidence_fractions.shape[0], total_confidence_fractions, 'iterations', 'confidence fractions', 'total confidence fractions')
    return model_FineTune, total_losses, total_confidence_fractions

#===================================
# torch utils
#===================================
# torch Dataset, work with DataLoader to return batch for evaluation
class get_torch_dataset(Dataset):
    '''
    training_data, with labels:  n x [[xsize, ysize, 3], label], shape: (n, 2)
                   no   labels: [n, xsize, ysize, 3]
    '''
    def __init__(self, training_data):
        self.X = training_data
        self.n_samples = training_data.shape[0]
        # we handle the preprocess here
        self.preprocess = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if training_data[0].shape[0] == 2:
            x = self.preprocess(self.X[index][0])
        else:
            x = self.preprocess(self.X[index])
        return x
        
def run_thru_NN(model, dataloader, mode):
    '''
    for model_CLR or model_CLR_CLS, there are embeddings and projections. model_effnet returns only embeddings.
    call mode = 'emb', 'proj' or others.
    '''
    output = []
    for _, batch in enumerate(dataloader):
        with torch.no_grad():
            model.eval().cuda()
            if mode == 'emb':
                output_batch, _ = model(batch.cuda())
            elif mode == 'proj':
                _, output_batch = model(batch.cuda())
            else:
                output_batch = model(batch.cuda())
        output.append(output_batch)
    output = torch.cat(output, dim=0)
    return output
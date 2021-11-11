#===================================
# Global variables
#===================================
# 0. color code               
color_list_35 = ['black','gray','silver', 'rosybrown', 'lightcoral','firebrick','maroon', 'red','coral','chocolate',
                 'saddlebrown','darkorange','orange','peachpuff','gold','olive','olivedrab','chartreuse','limegreen',
                 'darkgreen','darkslategray','teal','cyan','deepskyblue','dodgerblue','steelblue','navy',
                 'blue','slateblue','darkviolet','violet','purple','fuchsia','deeppink','pink']

# 1. timezone
tz = timezone('EST')

#===================================
# DEFs
# 0, organize PIL dataset
#===================================
def get_PIL_set(raw_data):
    '''
    raw_data is numpy array with dims: [n * [ [img_size, img_size, 3], label ] ]  -> shape: (n, 2)
                                       or [n, img_size, img_size, 3]
    '''
    transform = transforms.Compose([transforms.ToPILImage()])

    PIL_set = []
    labels = []
    for i in range(raw_data.shape[0]):
        data = raw_data[i]
        # if there's label in raw_data, raw_data.shape would be (n, 2)
        # otherwise, likely to be (n, img_size, img_size, 3)
        if len(raw_data.shape) == 2:
            img_pil = transform(raw_data[i][0])
            label = raw_data[i][1]               # true labels
        else:
            img_pil = transform(data)
            label = 0                                 # pseudo labels as 0
        PIL_set.append(img_pil)
        labels.append(label)
    labels = np.array(labels).reshape((-1, 1))
    return PIL_set, labels
    
#===================================
# DEFs
# 1 - 3, plots
#===================================
# 1. 2D embedding visualization
def plot_2d_scatter(X, Y, title='2d scatter plot', ColorList = color_list_35):
    '''
    return an interactive, color, size and hover data could be customized.
    '''
    # prepare idx
    idx = np.linspace(0, X.shape[0]-1, X.shape[0]).astype(int).reshape((X.shape[0], 1))
    # prepare X
    X_round = np.round_(X, 2)

    emb = np.hstack((X_round, Y, idx)) # messed up decimals
    emb = np.round_(emb, 2)
    # organize my own df
    df = pd.DataFrame(emb, columns = ['emb1', 'emb2', 'labels', 'index'])
    # sort on labels to ensure the color labeling starts from 0 to 30
    df = df.sort_values(by=['labels'])
    # take care the formating
    df['labels'] = df['labels'].astype(np.int).astype(str)

    # px plot
    fig = px.scatter(df, x = 'emb1', y = 'emb2', title = title,
                        color='labels', opacity=0.5, color_discrete_sequence=ColorList,
                        hover_data=['labels', 'emb1', 'emb2', 'index'])

    # other settings:
    fig.update_layout(width = 800, height = 500)
    fig.update_xaxes(scaleanchor = "y", scaleratio = 1)

    fig.show()
    
def save_2d_scatter(features, labels, title, savedir):
    # take care color map
    n_classes = np.unique(labels).shape[0]
    ColorList = matplotlib.colors.LinearSegmentedColormap.from_list("", color_list_35[:n_classes])
    # plot
    plt.figure(figsize=(10,10))
    plt.scatter(features[:,0], features[:,1], s=100, c=labels, cmap = ColorList, alpha = 0.5)
    plt.xlabel('emb1', fontsize = 20)
    plt.ylabel('emb2', fontsize = 20)
    plt.title(title, fontsize = 30)
    # save
    plt.savefig(savedir, dpi=300)
    plt.close()

# 2. plot img
def get_imgs(img_list, data):
    '''
    include [index1, index2] in a list. 
    data is the npy loaded numpy array including labels. 
    '''
    for i in img_list:
        if len(data.shape) == 2:
            plt.imshow(data[i][0])
            plt.title('label: ' + str(data[i][1]) + ', index: ' +str(i))
            plt.show() 
        else:
            plt.imshow(data[i])
            plt.title('index: ' +str(i))
            plt.show() 

# 3. plot for training process
def plot_losses(n_iterations, loss, xlabel, ylabel, title, savedir = None):
    '''
    plot training records (loss/confident fraction...) during iterations.
    to show / save
    '''
    iterations = np.linspace(1, n_iterations, n_iterations).astype(int)
    # plot
    plt.figure(figsize=(10, 10))
    plt.plot(iterations, loss)
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    # show/save
    if savedir is not None:
        plt.savefig(savedir, dpi=300)
    else:
        plt.show()
    plt.close()
#===================================
# DEFs
# 4 - 6, evaluation
#===================================

# 4. count number of samples in each group. Only for training data with labels. 
def get_CountsPerGroup(training_data):
    CountsPerGroup = np.zeros((training_data.shape[0])).astype(int)                            
    for i in range(training_data.shape[0]):
        label = training_data[i][1]
        CountsPerGroup[label] += 1
    CountsPerGroup = CountsPerGroup[: np.nonzero(CountsPerGroup)[0][-1]+1] # take only nonzeros
    return CountsPerGroup

# 5. mine nearest neighbors from embeddings
def mine_nearest_neighbors(features, topk, labels = None):
    '''
    min nearest neighbors with cosine distance 
    input:
    features: normalized (F.normalize(X_CLR_feat, dim = 1)) numpy array, shape=(n_samples, dim_features)
    topk: n_neighbors to mine. In our case, try 1 or 3.
    labels: groud truth labels shape = (n_samples, 1)

    output:
    indices: wrt orders in training_data
             numpy array, shape=(n_smaples, 1(anchor) + topk (neighbors))
    acc: mean accuracy of the neighbor assignments. 
    '''                                                                                    
    n, dim = features.shape[0], features.shape[1]

    # faiss to mine nearest neighbors
    index = faiss.IndexFlatIP(dim)                                                            
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, topk+1) # Sample itself is included

    if labels is None:
        return indices

    else:
        # compute the similarity in nearest neighbors
        neighbor_targets = np.take(labels, indices[:,1:], axis=0).reshape(n, topk) # Exclude sample itself for eval
        anchor_target = np.take(labels, indices[:, 0], axis=0)
        anchor_targets = np.repeat(anchor_target, topk, axis=1)
        acc = np.mean(neighbor_targets  == anchor_targets)
        return indices, acc

def get_knn_accs(features, labels, topks):
    '''
    faiss to mine knn, then check their assignments with anchor
    '''
    accs = []
    for k in topks:
        _, acc = mine_nearest_neighbors(features, k, labels) 
        accs.append(round(acc, 4))
    return accs


# 6. calculate accuracy for each group and wighted accuracy
def get_Acc(test_labels, gt_labels):
    '''
    only compute the acc.
    '''
    labels_uni, labels_count = np.unique(test_labels, return_counts=True)
    group_counts = []
    accs = []

    for i in range(labels_uni.shape[0]):
        
        # 1. for each cluster, get all imgs
        label = labels_uni[i]
        count = labels_count[i]
        idx = test_labels == label
        gt_label = gt_labels[idx]  #groud truth labels for imgs in a group
        
        # 3. compute acc in this group
        values, counts = np.unique(gt_label, return_counts=True)
        maj_vote = values[np.argmax(counts)]
        n_vote = np.sum(gt_label == maj_vote)
        acc = round(n_vote/gt_label.shape[0], 2)

        # 4. store data
        group_counts.append(count)
        accs.append(acc)

    # organize results
    group_counts = np.array(group_counts).reshape(-1, 1)
    accs = np.array(accs).reshape(-1, 1)
    group_info = np.hstack((group_counts, accs))
    # 1st data, may have nan scores when 1 img in 1 group
    df_group_info = pd.DataFrame(group_info, columns=['group_counts', 'acc'])
    group_weights = (df_group_info['group_counts']/np.sum(df_group_info['group_counts'])).values.copy()
    df_group_info['group_weights'] = group_weights
    final_score = np.round(np.sum(group_weights*df_group_info.values[:,1], axis=0), 4)
    return final_score, df_group_info

#===================================
# DEFs
# 7 - 9, output results
#===================================

# 7. outputs results in a folder
def output_folder(imgs, mzs, labels, cmap, AspectRatio, foldername, try_mode=False, probs=None):
    '''
    output results into folders at a designated directory in Google colab enviornment

    imgs: np.array in shape of (n_imgs, height, width)
    labels: np.array in shape of (n_imgs, 1) or (n_imgs)
    cmap: 'hot'  -> for plotting original imgs, 'gray' -> for plotting resized training imgs
    AspectRatio: usually 1 for 'gray', 15 for 'hot'
    foldername: dump to default directory, simply 'description_' + 'n_clusters'

    try_mode: only save 1 img, intends to test plotting setting
    probs: confidence score from softmax
    '''
    moment = datetime.now(tz) 
    moment = str(moment.date()) + '_' + str(moment.hour) + '-' + str(moment.minute)
    folder_dir = foldername + '_' + moment
    labels = labels.flatten()
    mzs = mzs.flatten()

    ### organize labels
    #===================================
    uni_labels, uni_counts = np.unique(labels, return_counts=True)         # ! inputs
    print('uni_labels: \n{}'.format(uni_labels))
    print('uni_counts: \n{}'.format(uni_counts))

    ### make folders for results
    #===================================
    subdirs = []

    # make the folder
    os.mkdir(folder_dir)

    # make subfolders
    print('\nMaking folders for results:\n')
    for i in tqdm(range(uni_labels.shape[0])):
        try:
            title = str(uni_labels[i]) + '_' + str(uni_counts[i])
            subdir = os.path.join(folder_dir, title)

            os.mkdir(subdir)
            subdirs.append(subdir)
        except:
            pass

    ### generate imgs in folders
    #===================================
    print('\nGenerating images in folders :\n')
    for i in tqdm(range(imgs.shape[0])):
        # get info
        img = imgs[i]
        label = labels[i]
        folder_idx = np.where(uni_labels == label)[0][0]

        # img dir
        if probs is not None:
            img_dir = subdirs[folder_idx] + '/' + str(i) + '__' + 'label_' + str(label) + '__mz_' + str(mzs[i]) + '__' + 'E_' + str(round(probs[i], 2)) + '.png'
        else:
            img_dir = subdirs[folder_idx] + '/' + str(i) + '__' + 'label_' + str(label) + '__mz_' + str(mzs[i]) + '.png'

        # plot
        if cmap == 'gray':
            plt.imshow(img, cmap=cmap, aspect=AspectRatio)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(img_dir)
            plt.close()
            plt.clf()
        elif cmap == 'hot':
            thre = np.quantile(img, 0.999)  # assume we didn't do any outlier handeling for org_imgs
            plt.imshow(img, cmap=cmap, vmax = thre, aspect=AspectRatio)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(img_dir)
            plt.close()
            plt.clf()
        
        if try_mode:
            break

    # zip up, ! commend doesn't work in this manner...
    #dir_zip = folder_dir + '.zip'
    #dir_folder = folder_dir + '/'
    #!zip -r $dir_zip $dir_folder
    print('all done')
    
# 8. output results as tightly packed montage images.
def output_montage_tight(imgs, mzs, cmap, img_AspectRatio, pixel_AspectRatio, labels=None, try_mode=False, SaveDir='montage_tight.png'):
    '''
    inputs: 1. imgs: (n, NumLine, NumSpePerLine)
            2. mzs: (n, )
            3. labels: (n, )
    cmap: 'hot' -> org_img with thresholding, 'gray' -> resized training img
    img_AspectRatio: 0-1
    pixel_AspectRatio: need to try for 'hot', 1 for 'gray' 
    SaveDir = save in default dir or designated folder
    '''
    # parameters:
    w_fig = 20 # default
    n_cols = 15 # default
    # data process
    mzs = mzs.flatten()
    if labels is not None:
        labels = labels.flatten()

    n_rows = math.ceil((imgs.shape[0])/n_cols)
    h_fig = w_fig * n_rows * (img_AspectRatio + 0.2) / n_cols # 0.2 is the space for title parameters

    fig = plt.figure(figsize=(w_fig,h_fig))
    fig.subplots_adjust(hspace= -0.2, wspace=0)
    print('workong on the montage plotting:')
    for i in tqdm(range(imgs.shape[0])):
        img = imgs[i]
        if labels is not None:
            title = str(mzs[i]) + '  ' + str(labels[i])
        else:
            title = str(mzs[i])
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        if cmap == 'hot':
            thre = np.quantile(img, 0.999)
            ax.imshow(img, cmap=cmap, vmax=thre, aspect = pixel_AspectRatio, interpolation='none')
        elif cmap == 'gray':
            ax.imshow(img, cmap=cmap, aspect = pixel_AspectRatio, interpolation='none')
        
        ax.set_xticks([])
        ax.set_yticks([])
        # title
        ax.set_title(title, pad=3, fontsize = 7)
        
        if try_mode:
            break
        
    plt.savefig(SaveDir, dpi=150)
    plt.close()
    print('all done')
    
# 9. output results as split montage images.
def output_montage_split(imgs, mzs, labels, cmap, img_AspectRatio, pixel_AspectRatio, try_mode=False, SaveDir='montage_split.png'):
    '''
    inputs: 1. imgs: (n, NumLine, NumSpePerLine)
            2. mzs: (n, )
            3. labels: (n, )
    cmap: 'hot' -> org_img with thresholding, 'gray' -> resized training img
    img_AspectRatio: 0-1
    pixel_AspectRatio: need to try for 'hot', 1 for 'gray' 
    SaveDir = save in default dir or designated folder
    '''
    # parameters:
    w_fig = 20 # default
    n_cols = 15 # default
    # data process
    mzs = mzs.flatten()
    labels = labels.flatten()
    
    # figure out classes
    classes = np.unique(labels)
    n_rows = 0              # total rows for the plot
    start_rows = []         # start row index
    for i in range(classes.shape[0]):
        # figure out n_samples for a lebel
        label = classes[i]
        n_samples = np.where(labels == label)[0].shape[0]

        # figure out n_row for label
        start_rows.append(n_rows)
        n_row = math.ceil(n_samples/n_cols)
        n_rows += n_row

    # plot imgs in different labels
    h_fig = w_fig * n_rows * (img_AspectRatio + 0.2) / n_cols # 0.2 is the space for title parameters
    fig = plt.figure(figsize=(w_fig,h_fig))
    fig.subplots_adjust(hspace= -0.2, wspace=0)

    # split labels
    print('workong on the montage plotting:')
    for i in tqdm(range(classes.shape[0])):
        label = classes[i]
        indices = np.where(labels == label)[0]
        # plot each ion img within group
        for j in range(indices.shape[0]):
            img = imgs[indices[j]]
            title = str(mzs[indices[j]]) + '  ' + str(labels[indices[j]])
            ax = fig.add_subplot(n_rows, n_cols, start_rows[i]*n_cols+j+1)

            if cmap == 'hot':
                thre = np.quantile(img, 0.999)
                ax.imshow(img, cmap=cmap, vmax=thre, aspect = pixel_AspectRatio, interpolation='none')
            elif cmap == 'gray':
                ax.imshow(img, cmap=cmap, aspect = pixel_AspectRatio, interpolation='none')

            ax.set_xticks([])
            ax.set_yticks([])
            # title
            ax.set_title(title, pad=3, fontsize = 7)
            
            if try_mode:
                break
    plt.savefig(SaveDir, dpi=150)
    plt.close()
    print('all done')
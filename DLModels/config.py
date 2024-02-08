class CNNConfig:
    """CNN模型参数"""
    emb_size = 300
    num_filters = 2
    window_sizes = [3, 4, 5]


class CNNTrainingConfig:
    """设置CNN模型训练的参数"""
    learning_rate = 0.0015
    epoches = 8
    print_step = 64
    # ReduceLROnPlateau参数
    factor = 0.3
    patience = 1
    verbose = True

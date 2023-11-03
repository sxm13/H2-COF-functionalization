import torch
from model.CGCNN_data_pre import CIFData,get_train_val_test_loader,collate_pool
import pandas as pd
from torch.autograd import Variable
from model.CGCNN_model import CrystalGraphConvNet,Normalizer
import numpy
from random import sample

def _get_device():
        if torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.set_device(0)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

def predict(train_csv, test_csv, root_dir, model_path, all_csv):
    device = _get_device()
    dataset_all  = CIFData(root_dir=root_dir, data_file=all_csv)
    sample_data_list = [dataset_all[i] for i in range(len(dataset_all))]
    _, sample_target, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)

    dataset_train = CIFData(root_dir=root_dir, data_file=train_csv)
    structures, _, _ = dataset_train[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,n_conv = 3,n_out=15)
    model.load_state_dict(torch.load(model_path + "checkpoints/model.pth", map_location=torch.device('cpu')))
    model.eval() 
    data_loader_train, _, _ = get_train_val_test_loader(
        dataset=dataset_train,
        random_seed=1029,
        collate_fn=collate_pool,
        pin_memory=False,
        batch_size=1,
        val_ratio=0,
        test_ratio=0
    )

    train_pre = []

    for bn, (input, target, name) in enumerate(data_loader_train):
        input_var = (
                Variable(input[0]),
                Variable(input[1]),
                input[2],
                input[3]
        )
        print(name)
        output_train = model(*input_var)
        output_train = normalizer.denorm(output_train.data.cpu())
   
        data_each = [name]
        for j in range(len(target.detach().numpy()[0])):
            data_each.extend([target.detach().numpy()[0][j],output_train.detach().numpy()[0][j]])
        train_pre.append(data_each)
    df_data_train = pd.DataFrame(train_pre,columns=["name","out1","pre1"
                                                    ,"out2","pre2"
                                                    ,"out3","pre3"
                                                    ,"out4","pre4"
                                                    ,"out5","pre5"
                                                    ,"out6","pre6"
                                                    ,"out7","pre7"
                                                    ,"out8","pre8"
                                                    ,"out9","pre9"
                                                    ,"out10","pre10"
                                                    ,"out11","pre11"
                                                    ,"out12","pre12"
                                                    ,"out13","pre13"
                                                    ,"out14","pre14"
                                                    ,"out15","pre15"])
    df_data_train.to_csv(model_path + "train_compare.csv")

    dataset_test = CIFData(root_dir=root_dir, data_file=test_csv)
    structures, _, _ = dataset_test[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,n_conv = 3,n_out=15)
    model.load_state_dict(torch.load(model_path + "checkpoints/model.pth", map_location=torch.device('cpu')))
    model.eval() 
    data_loader_test, _, _ = get_train_val_test_loader(
        dataset=dataset_test,
        random_seed=1029,
        collate_fn=collate_pool,
        pin_memory=False,
        batch_size=1,
        val_ratio=0,
        test_ratio=0
    )

    test_pre = []
    for bn, (input, target, name) in enumerate(data_loader_test):
        input_var = (
                Variable(input[0]),
                Variable(input[1]),
                input[2],
                input[3]
        )
        output_test = model(*input_var)
        output_test = normalizer.denorm(output_test.data.cpu())

        print(name)
        data_each = [name]
        for j in range(len(target.detach().numpy()[0])):
            data_each.extend([target.detach().numpy()[0][j],output_test.detach().numpy()[0][j]])
        test_pre.append(data_each)
    df_data_test = pd.DataFrame(test_pre,columns=["name","out1","pre1"
                                                    ,"out2","pre2"
                                                    ,"out3","pre3"
                                                    ,"out4","pre4"
                                                    ,"out5","pre5"
                                                    ,"out6","pre6"
                                                    ,"out7","pre7"
                                                    ,"out8","pre8"
                                                    ,"out9","pre9"
                                                    ,"out10","pre10"
                                                    ,"out11","pre11"
                                                    ,"out12","pre12"
                                                    ,"out13","pre13"
                                                    ,"out14","pre14"
                                                    ,"out15","pre15"])
    df_data_test.to_csv(model_path + "test_compare.csv")

import torch
from model.CGCNN_data_pre import CIFData,get_train_val_test_loader,collate_pool
import pandas as pd
from torch.autograd import Variable
from model.CGCNN_model import CrystalGraphConvNet,Normalizer
import numpy
from random import sample

def _get_device():
        if torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.set_device(0)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

def predict_all(train_csv, test_csv, root_dir, model_path, all_csv):
    device = _get_device()
    dataset_all  = CIFData(root_dir=root_dir, data_file=all_csv)
    sample_data_list = [dataset_all[i] for i in range(len(dataset_all))]
    _, sample_target, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)

    dataset_train = CIFData(root_dir=root_dir, data_file=train_csv)
    structures, _, _ = dataset_train[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,n_conv = 3,n_out=3)
    model.load_state_dict(torch.load(model_path + "checkpoints/model.pth", map_location=torch.device('cpu')))
    model.eval() 
    data_loader_train, _, _ = get_train_val_test_loader(
        dataset=dataset_train,
        random_seed=1029,
        collate_fn=collate_pool,
        pin_memory=False,
        batch_size=1,
        val_ratio=0,
        test_ratio=0
    )

    train_pre = []

    for bn, (input, target, name) in enumerate(data_loader_train):
        input_var = (
                Variable(input[0]),
                Variable(input[1]),
                input[2],
                input[3]
        )
        print(name)
        output_train = model(*input_var)
        output_train = normalizer.denorm(output_train.data.cpu())
   
        data_each = [name]
        for j in range(len(target.detach().numpy()[0])):
            data_each.extend([target.detach().numpy()[0][j],output_train.detach().numpy()[0][j]])
        train_pre.append(data_each)
    df_data_train = pd.DataFrame(train_pre,columns=["name","out1","pre1"
                                                    ,"out2","pre2"
                                                    ,"out3","pre3"
                                                    ])
    df_data_train.to_csv(model_path + "train_compare.csv")

    dataset_test = CIFData(root_dir=root_dir, data_file=test_csv)
    structures, _, _ = dataset_test[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,n_conv = 3,n_out=3)
    model.load_state_dict(torch.load(model_path + "checkpoints/model.pth", map_location=torch.device('cpu')))
    model.eval() 
    data_loader_test, _, _ = get_train_val_test_loader(
        dataset=dataset_test,
        random_seed=1029,
        collate_fn=collate_pool,
        pin_memory=False,
        batch_size=1,
        val_ratio=0,
        test_ratio=0
    )

    test_pre = []
    for bn, (input, target, name) in enumerate(data_loader_test):
        input_var = (
                Variable(input[0]),
                Variable(input[1]),
                input[2],
                input[3]
        )
        output_test = model(*input_var)
        output_test = normalizer.denorm(output_test.data.cpu())

        print(name)
        data_each = [name]
        for j in range(len(target.detach().numpy()[0])):
            data_each.extend([target.detach().numpy()[0][j],output_test.detach().numpy()[0][j]])
        test_pre.append(data_each)
    df_data_test = pd.DataFrame(test_pre,columns=["name","out1","pre1"
                                                    ,"out2","pre2"
                                                    ,"out3","pre3"
                                                    ])
    df_data_test.to_csv(model_path + "test_compare.csv")


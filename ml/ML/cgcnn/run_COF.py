from model.CGCNN_run import FineTune

cgcnn_run = FineTune(epoch = 500, lr = 0.001, batch_size = 32, opti ="SGD", weight_decay = 0.0,dataset= "./data/COF_H2-all.csv",
                     root_dir = "./data/COF-all/", log_dir = "./results/each/",n_conv=3, n_out = 3)

#cgcnn_run = FineTune(epoch = 500, lr = 0.001, batch_size = 32, opti ="SGD", weight_decay = 0.0,dataset= "./data/COF_H2.csv",
#                     root_dir = "./data/COF/", log_dir = "./results/ori/",n_conv=3, n_out = 15)

cgcnn_run.train()
loss, metric = cgcnn_run.test()

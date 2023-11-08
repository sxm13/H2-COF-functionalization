from model.CGCNN_pre import predict,predict_all

# predict(train_csv = "./results/ori/train_names.csv", test_csv= "./results/ori/test_names.csv", root_dir ="./data/COF/",model_path = "./results/ori/",all_csv = "./data/COF_H2.csv")
predict_all(train_csv = "./results/each/train_names.csv",test_csv= "./results/each/test_names.csv",root_dir = "./data/COF-all/",model_path = "./results/each/",all_csv = "./data/COF_H2-all.csv")

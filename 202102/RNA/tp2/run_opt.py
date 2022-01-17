from pathlib import Path
import pandas as pd
import clust
import argparse
from sklearn.model_selection import train_test_split


def get_dataset(dataset):
    X = pd.read_csv(Path(f"./datasets/split/{dataset}_fs.csv"))
    y = pd.read_csv(Path(f"./datasets/split/{dataset}_target_fs.csv"))

    X['TARGET_AUX'] = y.values
    
    return X, "TARGET_AUX"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default = "madelon")
    parser.add_argument("-k", "--metrics", type=str, default = "Calinski-Harabasz")
    parser.add_argument("-b", "--binary", type=int, default = 1)
    parser.add_argument("-n", "--ngen", type=int, default = 20)
    parser.add_argument("-p", "--pop", type=int, default = 5)
    parser.add_argument("-c", "--cxpb", type=float, default = 0.7)
    parser.add_argument("-m", "--mutpb", type=float, default = 0.2)
    parser.add_argument("-i", "--indpb", type=float, default = 0.05)
    parser.add_argument("-t", "--threads", type=int, default = 1)
    
    args = vars(parser.parse_args())

    ## Falta knn pra todos, dunn pra gisette pra frente

    if args['dataset'] == "all":
        datasets = ["christensen"]
        # datasets = ["gisette", "gina_prior", "christensen"]
        # datasets = ["madelon", "scene", "gisette", "gina_prior", "christensen"]
    else:
        datasets = [args['dataset']]

    if args['metrics'] == "all":
        # methods = ["Calinski-Harabasz", "Davies-Bouldin", "Silhouette", "Index-C", "Dunn"]#, "KNN"]
        # methods = ["Calinski-Harabasz", "Davies-Bouldin", "Silhouette", "Dunn"]#, "KNN"]
        methods = ["Calinski-Harabasz", "Davies-Bouldin", "Silhouette"]
    else:
        methods = [args['metrics']]

    print("Running Genetic Feature Optimization")
    print(args)

    for ds in datasets:
        df, target = get_dataset(ds)
        for metric in methods:
            if ds == 'madelon' and metric == "Calinski-Harabasz":
                continue
            if ds == 'madelon' and metric == "Davies-Bouldin":
                continue
            if ds == 'madelon' and metric == "Silhouette":
                continue

            print(f"{ds} - {metric}")
            metrics = metric.split(",")
            fs = clust.GeneticFeatureSelection(df = df, target_feature = target, experiment_name = 'testing', metrics = metrics, categorical_features = None, binary = args['binary'] == 1, seed = 42)
            try:
                fs.fit(ngen = args['ngen'], pop = args['pop'], cxpb = args['cxpb'], mutpb = args['mutpb'], indpb = args['indpb'], processes = args['threads'])
            except Exception as e:
                print(f"Error: {ds} - {metric}")
                print(e)
                continue
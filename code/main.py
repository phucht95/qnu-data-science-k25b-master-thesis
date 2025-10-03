import pandas as pd
from sklearn.metrics import accuracy_score
from data_loader.clowns import Clowns
from data_loader.cosexp import CosExp
from data_loader.pima import Pima
from data_loader.ionosphere import Ionosphere
from data_loader.parkinsons import Parkinsons
from data_loader.breast import Breast
from models.ls_svm import LSSVM
from models.c_svm import CSVM
from models.adaboost import AdaBoost
from models.pwl_ls_svm import PWLLSSVM
from models.pwl_c_svm import PWLCSVM


DATASETS = [
    Clowns(train_size=500, test_size=500),
    CosExp(train_size=500, test_size=500),
    Pima(train_size=384, test_size=384),
    Ionosphere(train_size=176, test_size=175),
    Parkinsons(train_size=98, test_size=97),
    Breast(train_size=350, test_size=349)
]

MODELS = [
    LSSVM,
    CSVM,
    AdaBoost,
    PWLLSSVM,
    PWLCSVM
]


def init_model(model, dataset, dataset_name):
    return model(dataset, dataset_name)


def save_result():
    results = []
    for dataset in DATASETS:
        row = {"Dataset": dataset.name}
        for model_cls in MODELS:
            try:
                if "PWL" in model_cls.__name__:
                    model = model_cls(dataset, M=10)
                else:
                    model = model_cls(dataset)
                model.run()

                row[f"{model_cls.__name__}_accuracy"] = round(model.accuracy, 4)
                # row[f"{model_cls.__name__}_time"] = model.runtime

            except Exception as e:
                print(f"❌ Lỗi với {model_cls.__name__} trên {dataset.name}: {e}")
                row[f"{model_cls.__name__}_accuracy"] = None
                # row[f"{model_cls.__name__}_time"] = None
        results.append(row)
    df = pd.DataFrame(results)
    print("\n📊 Bảng kết quả tổng hợp:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    save_result()
    # run_model_on_all_dataset(CSVM)
    # run_model_on_all_dataset(AdaBoost)
    # run_model_on_all_dataset(PWLCSVM)
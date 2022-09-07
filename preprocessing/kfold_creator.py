from sklearn.model_selection import KFold as ogkfold
import os


class Kfold:

    def __init__(self, kfold_dir: str, k: int):
        self.kfold_dir = kfold_dir
        self.k = k


    def __call__(self, database: str, *args, **kwargs):

        kfold = ogkfold(self.k, shuffle=True, random_state=12345)
        db_list = os.listdir(database)
        db_list = [x for x in db_list if x.endswith(".pdb")]
        for fold, (train_ids, test_ids) in enumerate(kfold.split(db_list)):
            kfold_file_path = os.path.join(self.kfold_dir, f"fold_{fold}.txt")
            with open(kfold_file_path, "w") as f:
                f.write("Training Examples: \n")
                for i in train_ids:
                    f.write(db_list[i] + '\n')
                f.write("Test Examples: \n")
                for j in test_ids:
                    f.write(db_list[j] + '\n')
                f.write("Validation Examples: \n") # just there for this all to work